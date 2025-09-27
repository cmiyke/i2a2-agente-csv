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
from io import StringIO

# Carrega a variável de ambiente (Chave da API)
load_dotenv()

# Inicialização no escopo global para evitar NameError
llm = None 

# Prefix para a criação do agente principal - create_and_run_agent
prefix_completo = (
    "Você é um especialista em análise de dados. Sua única ferramenta é 'python_repl_ast'. "
    "Sua missão é SEMPRE fornecer uma ANÁLISE DETALHADA e SEMPRE responder em Português do Brasil. "
    
    "O formato de saída é RIGOROSO: Action Input DEVE conter APENAS CÓDIGO PYTHON. "
    
    # ⚠️ INSTRUÇÃO DE OBRIGAÇÃO FINAL ⚠️
    "Você DEVE ASSUMIR que a variável 'df' (DataFrame) e 'pd' (Pandas) JÁ EXISTEM e estão disponíveis. "
    "SUA PRIMEIRA AÇÃO DEVE SER: Action: python_repl_ast\nAction Input: df.head() ou df.info() "
    "NUNCA TENTE import pandas, pd.read_csv() ou qualquer código de carregamento de arquivo."
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
        
        # 5. Salva o DF e os objetos para uso posterior
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
    
    initial_analysis_prompt = (
        "O DataFrame para análise já está carregado. Você não precisa executar nenhum código."
        "Sua tarefa é ler as informações estruturais e estatísticas abaixo e fornecer um "
        "resumo de Análise Exploratória de Dados (EDA) detalhado e em Português do Brasil. "
        "Foque na distribuição de dados, valores nulos, desbalanceamento da variável 'Class' e quaisquer anomalias. "
        
        "\n\n--- INFORMAÇÕES DO DATAFRAME (df) ---\n"
        f"ESTRUTURA (df.info()):\n{df_info_str}\n\n"
        f"ESTATÍSTICAS (df.describe()):\n{df_describe_str}\n"
        "-------------------------------------\n\n"
        "Gere a análise em Português agora, com no mínimo 200 palavras."
    )

    # --- 4. CONFIGURAR E EXECUTAR O AGENTE TEMPORÁRIO (SEM EXECUÇÃO DE CÓDIGO) ---
    
    # Ferramenta para satisfazer o requisito do Agente
    python_tool = PythonREPLTool()
    
    tools = [
        Tool(
            name="python_repl_ast",
            func=python_tool.run,
            description="Esta ferramenta não é necessária, o contexto já foi fornecido."
        )
    ]

    temp_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True
    )

    print("Iniciando a Análise Exploratória de Dados (EDA) para a memória...")
    analysis = temp_agent.run(initial_analysis_prompt)
    print("EDA concluída e salva na memória.")

    # --- 5. SALVAR NA MEMÓRIA (CORREÇÃO APLICADA AQUI) ---
    memory.save_context(
        inputs={"input": "Resumo da Análise Exploratória de Dados (EDA) do CSV"},
        outputs={"output": analysis}
    )
    
    return analysis

## A função AGORA RECEBE o objeto de arquivo (uploaded_file_object) E o LLM
#def initial_analysis_and_memory(uploaded_file_object, llm_instance) -> str:
#    """Gera o texto de memória de forma autônoma usando o LLM."""
#    
#    st.info("Gerando análise inicial autônoma do novo CSV. Isso pode levar alguns segundos...")
#    
#  
#    #Modificação da variável prompt para exigir detalhes numéricos e evitar respostas genéricas.
#    prompt = f"""
#    Execute uma Análise Exploratória de Dados (EDA) detalhada neste conjunto de dados.
#    Você **DEVE** executar código Python. Suas conclusões devem conter:
#    
#    1.  **Tipos de Dados**: O número total de colunas e uma lista das colunas numéricas (sem contar a primeira de índice).
#    2.  **Desbalanceamento**: Se houver uma coluna binária (0/1) chamada 'Class' ou similar, reporte a **porcentagem exata** da classe minoritária. Caso contrário, reporte a coluna mais desbalanceada.
#    3.  **Estatística Chave**: O **valor médio e o desvio padrão** da coluna chamada 'Amount' ou da **primeira coluna numérica** que não seja a de índice.
#    
#    A resposta deve ser um **texto de conclusão e síntese**, incorporando os valores numéricos exatos encontrados.
#    """
#
#    try:
#        # **CRUCIAL**: Reinicia o ponteiro antes de criar o agente, para que ele possa ler.
#        uploaded_file_object.seek(0) 
#
#        # 2. Cria o agente TEMPORÁRIO (passando o objeto de arquivo)
#        temp_agent = create_csv_agent(
#            llm_instance,
#            uploaded_file_object, # <<< USA O OBJETO DE ARQUIVO, NÃO A STRING DE CSV
#            verbose=False, # Mantenha o verbose em False aqui para evitar logs excessivos
#            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#            allow_dangerous_code=True,
#            prefix=(
#                "Você é um especialista em Análise Exploratória de Dados (EDA). "
#                "Sua **ÚNICA FERRAMENTA é 'python_repl_ast'**. "
#                "Sua missão é realizar a EDA e retornar um resumo **SINTETIZADO e DETALHADO em Português do Brasil**. "
#                
#                # <<< INSTRUÇÃO CRÍTICA DE FORMATAÇÃO >>>
#                "O Action Input DEVE conter **APENAS o código Python válido**, sem aspas, comentários ou texto adicional. "
#                "O Action Input é estritamente para o REPL Python."
#            ),
#            agent_executor_kwargs={"handle_parsing_errors": True},
#            max_iterations=20 # Pode ser necessário um valor ligeiramente maior para EDA complexa
#        )
#        
#        # 3. Executa a análise:
#        memoria_text = temp_agent.run(prompt)
#        return memoria_text
#        
#    except Exception as e:
#        return f"Falha catastrófica ao gerar análise inicial autônoma: {e}"
#

# --- Módulo 3: Criação e Execução do Agente LangChain ---


def create_and_run_agent(file_input, question, llm):
   
    # Reposicionar o ponteiro do arquivo para o início (essencial para Streamlit)
    file_input.seek(0) 
    
    # Carregar o DataFrame
    df = pd.read_csv(file_input) 
    
    # Crie a ferramenta (sem argumentos de escopo que falham)
    python_tool = PythonREPLTool() 

       
    tools = [
    Tool(
        name="python_repl_ast",
        func=python_tool.run,
        description=(
            "Uma ferramenta Python (REPL) que pode executar código Python para análise e gráficos. "
            "O DataFrame principal é a variável 'df' e a biblioteca Pandas é 'pd'. "
            "VOCÊ NUNCA PRECISA USAR 'pd.read_csv()'. O DataFrame JÁ ESTÁ CARREGADO NA VARIÁVEL 'df'."
        ),
    )
]
    #tools = [
    #    Tool(
    #        name="python_repl_ast",
    #        func=python_tool.run,
    #        description=(
    #            "Uma ferramenta Python (REPL) que pode executar código Python para manipulação de dados e gráficos. "
    #            # INSTRUÇÃO FINAL: Faça o agente usar a variável que está carregada no ambiente.
    #            "O DataFrame principal para TODAS as análises é a variável 'df'. Você DEVE iniciar todo script com 'import pandas as pd' e utilizar o 'df'."
    #        ),
    #    )
    #]

    # 3. Criar o Agente usando initialize_agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=15,
        
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
        
        # --- Lógica do Módulo de Memória ---
        if "conclusões" in user_question.lower() or "análise inicial" in user_question.lower():
            # Se for uma pergunta sobre a memória, usa a análise inicial preenchida
            st.info("Resposta baseada na Análise Inicial (Memória do Agente):")
            st.markdown(st.session_state['memoria_conclusoes'])
            
        # --- Lógica do Módulo de Execução de Código ---
        else:
            # Roda o agente LangChain para gerar e executar o código
            #response = create_and_run_agent(st.session_state['df'], user_question, llm)
            response = create_and_run_agent(st.session_state['uploaded_file_object'], user_question, llm) # << USE ISSO
            
            # Formata a resposta
            st.subheader("Resposta do Agente:")
            
            # Tenta exibir o gráfico gerado (se houver)
            # O LangChain salva o gráfico em um buffer temporário que precisamos capturar.
            # Este é um padrão comum em agentes de código.
            try:
                # O LangChain muitas vezes gera e exibe gráficos automaticamente.
                # Se precisar de um controle mais fino:
                # Crie uma pasta 'plots' e instrua a LLM a salvar lá.
                
                # Para simplificar, vamos instruir a LLM a gerar e o Streamlit a exibir o último plot.
                st.write(response) # Exibe o texto da resposta
                if plt.get_fignums():
                    # Se houver uma figura ativa após a execução do código, mostre-a
                    st.pyplot(plt.gcf())
                    
            except Exception as e:
                st.write(f"Resposta do Agente: {response}")
                st.error(f"Não foi possível exibir o gráfico. {e}")
                

else:
    st.info("Por favor, carregue um arquivo CSV e insira sua chave da API para começar.")

