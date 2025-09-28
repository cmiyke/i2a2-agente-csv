import streamlit as st
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI  # modelo 3.5-turbo-16k
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage # Novo import necessário
from io import StringIO
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any

# Carrega a variável de ambiente (Chave da API)
load_dotenv()

# Inicialização no escopo global para evitar NameError
llm = None 

# TEMP_PLOT_PATH deve ser uma constante global
TEMP_PLOT_PATH = "temp_plot.png" 

# --- Prefixo Completo para o Agente Principal ---
prefix_completo = (
    "Você é um especialista em análise de dados. Suas ferramentas são 'python_repl_ast' e 'buscar_memoria_EDA'. "
    "Sua missão é SEMPRE fornecer uma ANÁLISE DETALHADA e SEMPRE responder em Português do Brasil. "
    "A fonte primária e mais confiável de informação é a sua memória, acessada por 'buscar_memoria_EDA'."
    
    # ⚠️ REGRAS DE BUSCA E EXTRAÇÃO DE DADOS 
    
    # Regra 1: Ação para Resumo/Conclusões (Reforça a conversão da intenção)
    "1. SE A PERGUNTA DO USUÁRIO BUSCAR ANÁLISE INICIAL, CONCLUSÕES OU RESUMO, VOCÊ DEVE USAR A FERRAMENTA 'buscar_memoria_EDA' COM O **Action Input EXATO: 'Análise Exploratória Completa'**. Esta ação deve ser sua prioridade absoluta para essas perguntas."

    # Regra 2: Ação para Dados Específicos
    "2. Se a pergunta for sobre um dado específico (média, correlação, desvio padrão), o Action Input deve ser a **pergunta completa** (ex: 'Qual a correlação de V17 com Class?')."
    
    # ⚠️ REGRAS DE BUSCA E EXTRAÇÃO DE DADOS (Vamos focar na prioridade)
    # Regra 3 e 4: Como extrair
    "3. Após usar 'buscar_memoria_EDA', leia a 'Observation' e **extraia APENAS o número ou a informação solicitada na pergunta atual.**"
    "4. Para extrair correlação, procure o valor na linha da variável V(n) e na coluna 'Class' dentro da tabela de estatísticas da Observation."
    
    # ⚠️ REGRAS DE FORMATAÇÃO DE RESPOSTA (Resolve o problema de repetição)
    "5. Depois de concluir sua análise, **VOCÊ DEVE FINALIZAR O PROCESSO COM A TAG 'Final Answer:'** seguida da sua resposta completa. Nunca gere a resposta final detalhada apenas no THOUGHT."
    "6. Sua resposta final DEVE ser única e relevante para a pergunta mais recente do usuário."
    
    # ⚠️ REGRAS DE USO DE CÓDIGO (Regras de contorno do NameError, e AGORA PERMITINDO GRÁFICOS)
    
    # 7. Restrição mantida para evitar cálculo de estatísticas
    "7. NUNCA use a ferramenta 'python_repl_ast' para tentar calcular estatísticas ou buscar colunas. Para isso, use a memória (Regras 1 e 2)."
    
    # 8. Ação para GRÁFICOS (REATIVADA)
    "8. Se a pergunta for sobre um GRÁFICO (e.g., boxplot, histograma, scatter plot), utilize a ferramenta **'python_repl_ast'**."
    "O Action Input DEVE ser o código Python completo, usando 'df', 'matplotlib.pyplot as plt' e 'seaborn as sns'."
    "EXEMPLO DE ACTION INPUT: sns.boxplot(x='Class', y='Amount', data=df); plt.title('Boxplot de Amount por Class');"
)

# --- Configurações Iniciais e Layout do Streamlit ---

st.set_page_config(
    page_title="Agente Inteligente de EDA para Fraudes",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💳 Agente de Análise de Transações de Cartão de Crédito")
st.markdown("Este agente é especializado em **Análise Exploratória de Dados (EDA)**, com foco em conjuntos de dados desbalanceados e transformados via PCA.")

# Garante que o DataFrame persista durante as interações
if 'df' not in st.session_state:
    st.session_state['df'] = None
    st.session_state['memoria_conclusoes'] = ""
    
# --- Módulo 1: Carregamento de Dados (Para o Teste do Professor) ---
from langchain_core.messages import HumanMessage, AIMessage

# ⚠️ FUNÇÃO CHAVE PARA INJEÇÃO DE ESCOPO ⚠️
def execute_code_in_scope(code: str, df: pd.DataFrame) -> str:
    """
    Executa o código, injetando o df, plt, sns no escopo local.
    Salva o gráfico, se houver, e retorna o resultado da execução.
    """
    local_vars = {"df": df, "plt": plt, "sns": sns, "pd": pd}
    
    # 1. Limpa o ambiente antes de começar
    plt.close('all') 
    
    try:
        # A instrução 'exec' coloca o código no escopo de local_vars
        exec(code, local_vars)
        
        # 2. Verifica se houve plotagem (a lógica de salvamento da imagem)
        if plt.get_fignums():
            plt.gcf().tight_layout()
            plt.savefig(TEMP_PLOT_PATH) # Salva a imagem
            plt.close('all')
            return f"Gráfico gerado com sucesso e salvo em: {TEMP_PLOT_PATH}"
        else:
            # Retorna o valor de 'plt.show()' se houver (para código que não plota)
            # ou simplesmente uma mensagem de sucesso
            return "Código executado com sucesso."
            
    except Exception as e:
        plt.close('all')
        return f"Erro na execução do código Python: {e}"

def smart_memory_lookup_tool(query: str, llm, memory) -> str:
    """
    Ferramenta que usa um sub-LLM para buscar a resposta na memória e retornar apenas o texto ou o dado.
    
    - Se a query for 'Análise Exploratória Completa', extrai o bloco de conclusões.
    - Caso contrário, extrai um dado específico (média, correlação) da tabela.
    """
    
    # 1. Obtém o conteúdo completo da memória (o texto longo da EDA)
    memory_content = memory.buffer_as_str 
    
    # 2. Lógica para definir o prompt de busca interno (o 'sub-LLM')
    
    # ⚠️ CASO 1: EXTRAÇÃO DE RESUMO/CONCLUSÕES (Query Genérica)
    if "Análise Exploratória Completa" in query:
        search_prompt = (
            "Você é um Sumarizador de Conclusões. Seu trabalho é extrair a análise textual da Análise Exploratória de Dados (EDA) que está no CONTEÚDO abaixo."

            "\n\n--- INSTRUÇÃO DE EXTRAÇÃO ---\n"
            "1. **Procure o texto exato** entre as tags `[INÍCIO DO RESUMO ANALÍTICO]` e `[FIM DO RESUMO ANALÍTICO]`."
            "2. **Retorne APENAS o texto que está entre as tags**, sem incluir as tags em si."
            "3. Se as tags não forem encontradas, retorne a primeira análise textual completa que encontrar, mas NUNCA a tabela Markdown."

            f"\n\n--- CONTEÚDO DA EDA ---\n{memory_content}"
            "\n\nRetorne agora a análise completa:"
        )
        
    # ⚠️ CASO 2: EXTRAÇÃO DE DADOS ESPECÍFICOS (Query Específica)
    else:
        search_prompt = (
            "Você é um Extrator de Dados de Tabelas Markdown. Sua única função é encontrar e retornar um valor específico de uma tabela de estatísticas descritivas (df.describe())."
            
            f"\n\n--- PERGUNTA (Sua meta) ---\n{query}"
            
            "\n\n--- INSTRUÇÃO DE EXTRAÇÃO ---\n"
            "1. Procure na seção 'ESTATÍSTICAS COMPLETAS' do CONTEÚDO abaixo."
            "2. Identifique o valor exato (média, desvio, ou correlação) solicitado na PERGUNTA. Ex: V17 com Class."
            "3. **Retorne APENAS o número encontrado**, ou uma breve frase que o contenha (Ex: 'O valor é -0.326984')."
            "4. Se a informação não for encontrada na tabela, diga 'Informação indisponível para esta variável'."
            
            f"\n\n--- CONTEÚDO DA EDA ---\n{memory_content}"
            "\n\nExtraia o dado agora (NÃO GERE ANÁLISE LONGA):"
        )
        
    # 3. Executa o LLM para extrair a resposta
    try:
        # Nota: O método .invoke() ou .generate() depende da sua implementação exata do LangChain.
        # Estamos usando .invoke() para modelos recentes.
        extraction_message = llm.invoke(search_prompt)
        return extraction_message.content # Retorna apenas a extração limpa
        
    except Exception as e:
        # Isso será a Observation do Agente Principal, que ele pode usar para responder
        return f"Erro na extração de dados: {e}"


def memory_lookup_tool(query: str, memory: ConversationBufferMemory) -> str:
    """Ferramenta para buscar informações na análise exploratória de dados (EDA) da memória."""
    # A memória é injetada no agente, então a acessamos diretamente.
    
    # Simplesmente retornamos o buffer (o histórico de chat) como texto para que o LLM o analise.
    return memory.buffer_as_str
    
# --- ONDE VOCÊ CRIA A FERRAMENTA ---
def create_memory_tool(memory_instance: ConversationBufferMemory, llm_instance) -> Tool:
    """Retorna a ferramenta de busca de memória inteligente."""
    return Tool(
        name="buscar_memoria_EDA",
        # ⚠️ CHAVE: Passa a LLM e a memória para a nova função
        func=lambda q: smart_memory_lookup_tool(q, llm=llm_instance, memory=memory_instance),
        description=(
            "Use esta ferramenta EXCLUSIVAMENTE para buscar estatísticas (média, desvio, correlação) na Análise Exploratória de Dados (EDA). "
            "Retorna APENAS o dado solicitado, sem o texto completo da EDA."
        )
    )

# Modifique a função load_data para chamar a nova função:
def load_data(uploaded_file, llm_instance): 
    """Carrega o CSV, cria a memória, e delega a geração de análise inicial."""
    try:
        # 1. LÊ e salva o DF (apenas para exibição e a primeira leitura rápida)
        df = pd.read_csv(uploaded_file)
        
        # 2. Reinicia o ponteiro do arquivo.
        uploaded_file.seek(0)
        
        # 3. CRIA O OBJETO DE MEMÓRIA (NOVO PASSO!)
        # O agente principal e a função initial_analysis_and_memory precisam dele.
        memory_instance = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        
        # 4. CHAMA A FUNÇÃO DE MEMÓRIA (PASSANDO A MEMÓRIA!)
        # O objeto de arquivo é passado para que o DF possa ser lido dentro da função de análise.
        analysis_result = initial_analysis_and_memory(
            file_input=uploaded_file, 
            llm=llm_instance,
            memory=memory_instance # <<< CORREÇÃO AQUI!
        )
        
        # 5. Reinicia o ponteiro do arquivo NOVAMENTE para garantir que o AGENTE PRINCIPAL o encontre no Streamlit.
        # Mesmo que a função initial_analysis_and_memory tenha reiniciado, é uma boa prática
        # garantir o estado para o uso subsequente (eletivo, mas seguro).
        uploaded_file.seek(0) # ⚠️ Adicionado novamente por segurança
        
        # 6. Salva o DF e os objetos para uso posterior
        st.session_state['df'] = df
        st.session_state['uploaded_file_object'] = uploaded_file
        
        # O resultado da análise (analysis_result) pode ser o valor que você salva na session_state.
        # Ajustei o nome da chave para refletir o conteúdo (o próprio resultado, não a memória).
        st.session_state['analise_inicial'] = analysis_result
        
        # O objeto de memória deve ser salvo se for usado no agente principal interativo:
        st.session_state['memory_instance'] = memory_instance

        st.success("Arquivo carregado e análise inicial autônoma concluída! Pronto para perguntar.")
        
    except Exception as e:
        st.error(f"Erro fatal: {e}")

# --- Módulo 2: Geração de Conclusões Iniciais (A Memória) ---

# Crie uma versão simplificada do agente SEM a parte do Streamlit
def run_llm_analysis(df, prompt, llm):
    """Executa uma análise única e retorna a resposta."""
    # Nota: Você pode precisar redefinir o objeto de arquivo para o LangChain, 
    # mas para simplificar, usaremos o df diretamente (se o LangChain suportar, 
    # senão precisaremos salvar e ler o CSV temporariamente).
    
    # Vamos usar o Agente CSV que você já tem, mas com uma instrução mais curta.
    # OBS: Se você já tem a instância do LLM, reutilize-a!

    temp_agent = create_csv_agent(
        llm,
        df.to_csv(index=False), # Convertendo o DF para string CSV (objeto file-like)
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        # Instrução genérica para a análise inicial
        #prefix="Você deve executar uma Análise Exploratória de Dados (EDA) e retornar um resumo em texto. Não use o formato Thought/Action. Liste as colunas numéricas, estatísticas chave, e quaisquer colunas que pareçam desbalanceadas ou com alta correlação.",
        prefix="Você é um especialista em análise de dados. Sua única tarefa é executar o código Python necessário para analisar o CSV e **retornar o resumo em texto solicitado**. Não use o formato Thought/Action.",
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    try:
        # Pede ao LLM para fazer a análise de 5 minutos sobre o novo CSV
        return temp_agent.run(prompt)
    except Exception as e:
        return f"Falha ao gerar análise inicial autônoma: {e}"



# Função para realizar a análise exploratória e salvar na memória
def initial_analysis_and_memory(file_input, llm, memory):
    # --- 1. CARREGAR O DATAFRAME FORA DO AGENTE ---
    file_input.seek(0) 
    df = pd.read_csv(file_input) 

    # --- 2. GERAR O CONTEXTO MANUALMENTE (RESOLVENDO FALHA DE ESCOPO) ---
    
    # Capturar df.info() como string para a memória
    info_buffer = StringIO()
    df.info(buf=info_buffer)
    df_info_str = info_buffer.getvalue()

    # Capturar df.describe() como Markdown
    df_describe_str = df.describe(include='all').to_markdown()
    # --- 3. CONSTRUIR O PROMPT COM O CONTEXTO FORÇADO ---
    initial_analysis_prompt_text = (
        "Você é um analista de dados especialista e só fala Português do Brasil. "
        "Sua tarefa é ler as informações estruturais e estatísticas abaixo e fornecer um "
        "resumo de Análise Exploratória de Dados (EDA) detalhado, SEGUINDO O FORMATO SOLICITADO. "

        "\n\n--- INFORMAÇÕES DO DATAFRAME (df) ---\n"
        f"ESTRUTURA (df.info()):\n{df_info_str}\n\n"
        f"ESTATÍSTICAS (df.describe()):\n{df_describe_str}\n" 
        "-------------------------------------\n\n"

        # ⚠️ INSTRUÇÕES CRÍTICAS PARA O RESUMO TEXTUAL (Incluindo as Tags)
        "1. Comece sua resposta com a tag **[INÍCIO DO RESUMO ANALÍTICO]**."
        "2. Forneça a análise detalhada (resumo da EDA). A análise deve ser longa e completa, abordando todos os pontos do dataframe (estrutura, balanceamento da Class, variações de V-Columns, e outliers em Amount)."
        "3. Termine o resumo analítico com a tag **[FIM DO RESUMO ANALÍTICO]**."
        "4. APÓS a tag de fim, insira O TÍTULO 'ESTATÍSTICAS COMPLETAS' e, logo abaixo, a tabela de df.describe() no formato Markdown."
        
        "\n\n**Gere a análise AGORA, respeitando estritamente a ordem e as tags:**"
    )

    # --- 4. EXECUTAR O LLM DIRETAMENTE (SEM AGENTE) ---
    print("Iniciando a Análise Exploratória de Dados (EDA) para a memória (LLM Direto)...")
    
    # ⚠️ NOVO CÓDIGO CRÍTICO: Usar o LLM diretamente
    analysis_message = llm.invoke(initial_analysis_prompt_text)
    analysis = analysis_message.content # Extrai o texto da resposta
    
    print("EDA concluída e salva na memória.")

    # --- 5. SALVAR NA MEMÓRIA ---
    # Usamos o HumanMessage para simular a pergunta e o AIMessage para simular a resposta do assistente
    memory.save_context(
        inputs={"input": "Resumo da Análise Exploratória de Dados (EDA) do CSV"},
        outputs={"output": analysis}
    )
    
    return analysis


# --- Módulo 3: Criação e Execução do Agente LangChain ---

def create_and_run_agent(file_input, question, llm, memory_instance):
   
    # Se você não puder garantir que pd e df são visíveis globalmente, 
    # você pode forçar a declaração global, mas isso é opcional e depende da sua estrutura.
    global pd, df # Remova essa linha se ela estiver causando problemas
     
    # Reposicionar o ponteiro do arquivo para o início (essencial para Streamlit)
    file_input.seek(0) 
    
    # Carregar o DataFrame
    df = pd.read_csv(file_input) 
    
    # Crie a ferramenta (sem argumentos de escopo que falham)
    #python_tool = PythonREPLTool() #comentado para permitir geração de graficos

    # Crie a nova ferramenta de memória
    memory_tool = create_memory_tool(memory_instance, llm)
       
    tools = [
    Tool(
        name="python_repl_ast",
        func=lambda code: execute_code_in_scope(code, df=df),
            description=(
                "USE ESTA FERRAMENTA APENAS para gerar GRÁFICOS (boxplot, histograma, etc.). "
                "O DataFrame está disponível como 'df', 'seaborn' como 'sns', e 'matplotlib.pyplot' como 'plt'. "
                "A ferramenta automaticamente salva o gráfico. NUNCA use plt.show() ou plt.savefig()."
        ),
    ),
    memory_tool # Adiciona a ferramenta de memória
]

    # 3. Criar o Agente usando initialize_agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        memory=st.session_state['memory_instance'],
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=20,
        
        # Injetar o prefixo
        agent_kwargs={
            "prefix": prefix_completo
        },
        # Adicionar o handle_parsing_errors no executor
        handle_parsing_errors=True,
    )
    
    # 4. Executar o Agente
    response = agent.run(question)
    return response


# --- Sidebar e Interação do Usuário ---

with st.sidebar:
    st.header("Upload do Dataset")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    
    st.header("Configuração da LLM")
    api_key = os.getenv("OPENAI_API_KEY", "") 
    openai_api_key = st.text_input("Sua Chave OpenAI API", type="password", value=api_key)
    
    # 1. Definição da instância LLM
    if openai_api_key:
        # Nota: Você pode usar a otimização com st.cache_resource aqui!
        llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.0, model_name="gpt-3.5-turbo-16k") 
    else:
        llm = None
        
    # 2. Lógica de Carregamento de Dados (BLOCO 2)
    # Garante que o llm esteja definido antes de chamar load_data
    if uploaded_file is not None and st.session_state['df'] is None:
        if llm is not None:
            # =======================================================
            # <<< AQUI COMEÇA O BLOCO 2 (Chamada da Lógica de Dados) >>>
            load_data(uploaded_file, llm) # Note que 'llm' é passado como argumento
            # =======================================================
            # No seu arquivo principal, após a execução de load_data 
            # REMOVER ISSO DEPOIS DE TESTAR:
            if 'memory_instance' in st.session_state:
                print("Conteúdo da Memória:")
                # Isto irá imprimir o histórico de chat, que deve conter a EDA
                print(st.session_state['memory_instance'].buffer) 
        else:
            st.warning("Por favor, insira sua chave da OpenAI e carregue o CSV para inicializar o agente.")

    st.markdown("---")
    st.markdown("Requisitos do Trabalho:")
    st.markdown("- 4 Perguntas (sendo 1 gráfica)")
    st.markdown("- 1 Pergunta sobre as conclusões (Memória)")
    
# --- Interface Principal de Perguntas e Respostas ---

if st.session_state['df'] is not None and llm is not None:
    
    # Exemplo de perguntas para guiar o usuário e o professor
    st.subheader("Exemplos de Perguntas:")
    st.markdown("""
    1. **Estatística:** Qual o desvio padrão e a média da variável 'Time'?
    2. **Relação Gráfica:** Mostre um boxplot da variável 'Amount' para a classe de fraude (Class=1) versus a classe normal (Class=0).
    3. **Padrão:** Qual a correlação da variável V17 com a variável 'Class'?
    4. **Conclusões (Memória):** Quais as conclusões que você obteve a partir da análise inicial dos dados?
    """)
    
    user_question = st.text_input("Insira sua pergunta de EDA aqui:", key="user_input")

    if user_question:
        
        # Roda o agente LangChain para gerar e executar o código
        #response = create_and_run_agent(st.session_state['df'], user_question, llm)
        response = create_and_run_agent(st.session_state['uploaded_file_object'], user_question, llm, st.session_state['memory_instance']) # << USE ISSO
        
        # Formata a resposta
        st.subheader("Resposta do Agente:")
        st.markdown(response)
        
        # ⚠️ 3. LÓGICA DE EXIBIÇÃO DE GRÁFICO (AJUSTE CRÍTICO AQUI)

        TEMP_PLOT_PATH = "temp_plot.png" # Recria a constante para este escopo
        
        if os.path.exists(TEMP_PLOT_PATH):
            st.subheader("Visualização Gerada:")
            
            # Exibe a imagem salva no disco
            st.image(TEMP_PLOT_PATH)
            
            # Opcional: Remova o arquivo para que a próxima execução não pegue o gráfico antigo
            os.remove(TEMP_PLOT_PATH) 
else:
    st.info("Por favor, carregue um arquivo CSV e insira sua chave da API para começar.")

