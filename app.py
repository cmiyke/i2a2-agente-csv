# ======================================================================
# 1. IMPORTS E CONFIGURAÇÕES GLOBAIS
# ======================================================================

# Módulos de Interface e Estrutura
import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import os
import io
from io import StringIO
from typing import Dict, Any


# Módulos de Visualização e Manipulação de Dados
import matplotlib.pyplot as plt
import seaborn as sns

# Módulos LangChain e LLM
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# --- Configurações e Constantes ---
load_dotenv()
TEMP_PLOT_PATH = "temp_plot.png"
llm = None # Inicialização no escopo global

# ======================================================================
# 2. PREFIXO DO AGENTE (Regras de Comportamento)
# ======================================================================

prefix_completo = (
    "Você é um Agente de Análise de Dados (EDA) especialista em Python, com a tarefa de analisar o DataFrame 'df'."
    "Seu objetivo é ajudar o usuário a entender os dados e gerar gráficos."
    
    # 1. Regras Gerais
    "1. Responda em Português."
    "2. Utilize a ferramenta 'buscar_memoria_EDA' APENAS para responder a perguntas conceituais ou de resumo (ex: 'Quais as conclusões da análise?')."
    "3. Use a ferramenta 'python_repl_ast' APENAS para gerar GRÁFICOS ou realizar cálculos. **Para plots comparativos (boxplot), use APENAS a sintaxe: sns.boxplot(x='coluna_categorica', y='coluna_numerica', data=df).** Para correlação, use o cálculo: df['coluna1'].corr(df['coluna2'])."
    "4. Ao gerar gráficos, use apenas colunas relevantes do 'df'."
    "5. NUNCA mencione o uso da ferramenta, o código executado ou a 'Observation' na sua Resposta Final."

    # 2. Regras de Prioridade Analítica (Anti-Loop)
    "6. NUNCA repita uma 'Action' cuja 'Observation' não forneceu a informação necessária. Isso gera um loop."
    "7. Se a pergunta for sobre outlier ou exigir GRÁFICO, NÃO utilize a ferramenta 'buscar_memoria_EDA'."
    "7a. **FILTRO DE PRIORIDADE ANALÍTICA:** Se a pergunta exigir **cálculo ou evidência visual**, o Agente DEVE usar o **'python_repl_ast'** como a sua **PRIMEIRA AÇÃO**."

    # 3. Regras de Sintaxe e Parsing (Anti-Loop de Confirmação)
    "8. O seu ciclo de raciocínio DEVE **SEMPRE** usar os termos em **INGLÊS** e **ESTRITAMENTE** sem variações."
    "Use estritamente: **Thought:**, **Action:**, **Action Input:**, **Observation:**, e **Final Answer:**."
    "8.a. O valor da tag **'Action:' DEVE SER APENAS O NOME DA FERRAMENTA**."
    "8.b. Para 'python_repl_ast', o Action Input DEVE ser **código Python válido**."
    "8.c. O código do gráfico DEVE SEMPRE terminar com um comando de exibição, como `plt.show()`."
    "8.d. **ANTI-LOOP:** O **'Thought:' DEVE SER CURTO**, conciso e focado apenas no motivo da próxima **Action:** ou, após o sucesso, na **análise** para a **Final Answer:**. **NUNCA REPITA O OBJETIVO INICIAL OU A PERGUNTA APÓS UMA OBSERVATION DE SUCESSO**."

    # 4. Regra para Geração de Exemplos de Perguntas (SAÍDA DIRETA)
    "9. **PERGUNTAS DINÂMICAS (SAÍDA ÚNICA):** Se a solicitação for **GERAR EXEMPLOS DE PERGUNTAS** (após a análise inicial), o Agente **NÃO DEVE USAR NENHUMA FERRAMENTA**. A resposta deve ser formulada diretamente como uma **Final Answer** contendo **4 perguntas** em formato de lista Markdown, seguindo a ordem: 1) Conclusões Sobre a Análise de Dados Inicial, 2) Cálculo Estatístico 1, 3) Cálculo Estatístico 2, 4) Geração de Gráfico. **NUNCA CHAME UMA ACTION ANTES DA RESPOSTA FINAL NESTE CASO.**"
    
    # 5. Regra de Parada Única (Análise Final)
    "10. **ESTRUTURA DE PARADA CRÍTICA:** Após a 'Observation' final (principalmente após um gráfico), a próxima e ÚLTIMA saída DEVE ser em sequência direta e seguir **estritamente este formato**: Thought: [Sua análise CONCISA do gráfico ou dado obtido].\nFinal Answer: [Sua resposta concisa e analítica em português]."
    "NÃO USE QUALQUER TEXTO OU TAG APÓS O 'Final Answer:'."
)

# ======================================================================
# 3. FERRAMENTAS E FUNÇÕES DE AJUDA
# ======================================================================

# ----------------------------------------------------------------------
# FERRAMENTA PYTHON REPL (Execução e Plotagem)
# ----------------------------------------------------------------------

def execute_code_in_scope(code: str, df: pd.DataFrame) -> str:
    """
    Executa o código Python, injetando o df, plt, sns no escopo local.
    Salva o gráfico gerado (se houver) no arquivo TEMP_PLOT_PATH.
    
    Args:
        code (str): Bloco de código Python a ser executado.
        df (pd.DataFrame): DataFrame principal para manipulação.

    Returns:
        str: Mensagem de sucesso ou erro, incluindo o caminho do gráfico.
    """
    local_vars = {"df": df, "plt": plt, "sns": sns, "pd": pd}
    
    plt.close('all')  
    
    try:
        # A instrução 'exec' coloca o código no escopo de local_vars
        exec(code, local_vars)
        
        # Verifica se houve plotagem e salva o gráfico
        if plt.get_fignums():
            plt.gcf().tight_layout()
            plt.savefig(TEMP_PLOT_PATH)
            plt.close('all')
            return f"Gráfico gerado com sucesso e salvo em: {TEMP_PLOT_PATH}"
        else:
            return "Código executado com sucesso."
            
    except Exception as e:
        plt.close('all')
        return f"Erro na execução do código Python: {e}"

# ----------------------------------------------------------------------
# FERRAMENTA DE BUSCA NA MEMÓRIA (Sub-LLM para Extração)
# ----------------------------------------------------------------------

def smart_memory_lookup_tool(query: str, llm: ChatOpenAI, memory: ConversationBufferMemory) -> str:
    """
    Ferramenta que usa um sub-LLM para buscar a resposta na memória e retornar apenas 
    o texto específico (resumo) ou o dado solicitado (estatística).
    """
    memory_content = memory.buffer_as_str  
    
    # Lógica para definir o prompt de busca interno (o 'sub-LLM')
    if "Análise Exploratória Completa" in query or "conclusões" in query.lower():
        search_prompt = (
            "Você é um Sumarizador de Conclusões. Seu trabalho é extrair a análise textual da Análise Exploratória de Dados (EDA) que está no CONTEÚDO abaixo. "
            "Procure o texto exato entre as tags `[INÍCIO DO RESUMO ANALÍTICO]` e `[FIM DO RESUMO ANALÍTICO]` e retorne APENAS o texto entre elas. "
            f"\n\n--- CONTEÚDO DA EDA ---\n{memory_content}"
            "\n\nRetorne agora a análise completa:"
        )
    else:
        search_prompt = (
            "Você é um Extrator de Dados de Tabelas Markdown. Sua única função é encontrar e retornar um valor específico de uma tabela de estatísticas descritivas (df.describe())."
            f"\n\n--- PERGUNTA (Sua meta) ---\n{query}"
            "\n\n--- INSTRUÇÃO DE EXTRAÇÃO ---\n"
            "1. Procure na seção 'ESTATÍSTICAS COMPLETAS' do CONTEÚDO abaixo. "
            "2. Identifique o valor exato solicitado na PERGUNTA. "
            "3. Retorne APENAS o número encontrado, ou uma breve frase que o contenha. "
            f"\n\n--- CONTEÚDO DA EDA ---\n{memory_content}"
            "\n\nExtraia o dado agora (NÃO GERE ANÁLISE LONGA):"
        )
            
    try:
        extraction_message = llm.invoke(search_prompt)
        return extraction_message.content
            
    except Exception as e:
        return f"Erro na extração de dados: {e}"


def create_memory_tool(memory_instance: ConversationBufferMemory, llm_instance: ChatOpenAI) -> Tool:
    """Cria e retorna a ferramenta de busca de memória inteligente ('buscar_memoria_EDA')."""
    return Tool(
        name="buscar_memoria_EDA",
        func=lambda q: smart_memory_lookup_tool(q, llm=llm_instance, memory=memory_instance),
        description=(
            "Use esta ferramenta EXCLUSIVAMENTE para buscar estatísticas (média, desvio, correlação) ou "
            "o resumo completo na Análise Exploratória de Dados (EDA) previamente salva. "
            "Retorna APENAS o dado ou resumo solicitado, sem o texto completo da EDA."
        )
    )

# ----------------------------------------------------------------------
# ANÁLISE INICIAL AUTÔNOMA (Gera e Salva a Memória)
# ----------------------------------------------------------------------

def initial_analysis_and_memory(file_input: io.TextIOWrapper, llm: ChatOpenAI, memory: ConversationBufferMemory) -> str:
    """
    Realiza uma análise exploratória de dados (EDA) autônoma, gerando um resumo textual
    e salvando-o na memória para consulta posterior do agente principal.
    """
    delimiter = st.session_state.get('delimiter', ',')
    
    # 1. CARREGAR O DATAFRAME
    file_input.seek(0)  
    df = pd.read_csv(file_input, sep=delimiter)  

    # 2. GERAR O CONTEXTO ESTATÍSTICO MANUALMENTE
    info_buffer = StringIO()
    df.info(buf=info_buffer)
    df_info_str = info_buffer.getvalue()
    df_describe_str = df.describe(include='all').to_markdown()
    
    # 3. CONSTRUIR O PROMPT COM O CONTEXTO FORÇADO
    initial_analysis_prompt_text = (
        "Você é um analista de dados especialista e só fala Português do Brasil. "
        "Sua tarefa é ler as informações estruturais e estatísticas abaixo e fornecer um "
        "resumo de Análise Exploratória de Dados (EDA) detalhado, SEGUINDO O FORMATO SOLICITADO. "
        "\n\n--- INFORMAÇÕES DO DATAFRAME (df) ---\n"
        f"ESTRUTURA (df.info()):\n{df_info_str}\n\n"
        f"ESTATÍSTICAS (df.describe()):\n{df_describe_str}\n"  
        "-------------------------------------\n\n"
        "1. Comece sua resposta com a tag **[INÍCIO DO RESUMO ANALÍTICO]**."
        "2. Forneça a análise detalhada (resumo da EDA)."
        "3. Termine o resumo analítico com a tag **[FIM DO RESUMO ANALÍTICO]**."
        "4. APÓS a tag de fim, insira O TÍTULO 'ESTATÍSTICAS COMPLETAS' e, logo abaixo, a tabela de df.describe() no formato Markdown."
        "\n\n**Gere a análise AGORA, respeitando estritamente a ordem e as tags:**"
    )

    # 4. EXECUTAR O LLM DIRETAMENTE
    print("Iniciando a Análise Exploratória de Dados (EDA) para a memória (LLM Direto)...")
    analysis_message = llm.invoke(initial_analysis_prompt_text)
    analysis = analysis_message.content 
    print("EDA concluída e salva na memória.")

    # 5. SALVAR NA MEMÓRIA
    memory.save_context(
        inputs={"input": "Resumo da Análise Exploratória de Dados (EDA) do CSV"},
        outputs={"output": analysis}
    )
    
    return analysis

# ----------------------------------------------------------------------
# FUNÇÕES DE CARREGAMENTO E INICIALIZAÇÃO
# ----------------------------------------------------------------------

def load_data(uploaded_file: io.TextIOWrapper, llm_instance: ChatOpenAI): 
    """
    Orquestra o carregamento do CSV, inicialização da memória, execução da análise
    autônoma e salvamento dos objetos de estado na sessão do Streamlit.
    """
    delimiter = st.session_state.get('delimiter', ',')
    
    try:
        # 1. LÊ o DF e reinicia o ponteiro
        df = pd.read_csv(uploaded_file, sep=delimiter)
        uploaded_file.seek(0)
        
        # 2. CRIA O OBJETO DE MEMÓRIA
        memory_instance = ConversationBufferMemory(
            memory_key="chat_history",  
            return_messages=True
        )
        
        # 3. CHAMA A FUNÇÃO DE ANÁLISE INICIAL
        analysis_result = initial_analysis_and_memory(
            file_input=uploaded_file,  
            llm=llm_instance,
            memory=memory_instance 
        )
        
        # 4. Reinicia o ponteiro do arquivo para o Streamlit/Agente principal
        uploaded_file.seek(0) 
        
        # 5. Salva o DF e os objetos de estado
        st.session_state['df'] = df
        st.session_state['uploaded_file_object'] = uploaded_file
        st.session_state['analise_inicial'] = analysis_result
        st.session_state['memory_instance'] = memory_instance

        st.success("Arquivo carregado e análise inicial autônoma concluída! Pronto para perguntar.")
        
    except Exception as e:
        st.error(f"Erro ao carregar ou analisar os dados. Verifique o delimitador ou o formato do arquivo: {e}")

# ----------------------------------------------------------------------
# CRIAÇÃO E EXECUÇÃO DO AGENTE PRINCIPAL
# ----------------------------------------------------------------------

def create_and_run_agent(file_input: io.TextIOWrapper, question: str, llm: ChatOpenAI, memory_instance: ConversationBufferMemory) -> str:
    """
    Configura, inicializa e executa o agente LangChain com as ferramentas e regras definidas.
    """
    global df

    # Carregar o DataFrame novamente, garantindo que o agente use o estado correto
    file_input.seek(0)  
    df = pd.read_csv(file_input, sep=st.session_state.get('delimiter', ','))  

    # 1. CRIAÇÃO DAS FERRAMENTAS
    memory_tool = create_memory_tool(memory_instance, llm)
    
    tools = [
        Tool(
            name="python_repl_ast",
            func=lambda code: execute_code_in_scope(code, df=df),
            description=(
                "USE ESTA FERRAMENTA APENAS para gerar GRÁFICOS (boxplot, histograma, etc.) e realizar cálculos. "
                "O DataFrame está disponível como 'df', 'seaborn' como 'sns', e 'matplotlib.pyplot' como 'plt'. "
                "A função de execução salva o gráfico automaticamente no disco. NUNCA use plt.show() ou plt.savefig()."
            ),
        ),
        memory_tool
    ]

    # 2. INJEÇÃO DE INFORMAÇÕES NO PROMPT (Customização do Suffix)
    df_columns = ", ".join(df.columns.tolist())
    tool_names_string = ", ".join([t.name for t in tools])
    
    custom_instruction = (
        f"As colunas do DataFrame atual são: {df_columns}. "
        f"Você tem acesso às ferramentas: {tool_names_string}.\n\n"
    )

    react_memory_template = (
        "Histórico da Conversa: {chat_history}\n"
        "Histórico de Pensamento/Ação: {agent_scratchpad}\n"
        "Pergunta Atual: {input}\n\n"  
        "\nComece seu raciocínio (Thought):"  
    )
    final_suffix = custom_instruction + react_memory_template

    # 3. CRIAÇÃO E EXECUÇÃO DO AGENTE
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        memory=memory_instance,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=20,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": prefix_completo,
            "suffix": final_suffix,
            "input_variables": ["input", "chat_history", "agent_scratchpad"]
        },
    )
    
    response = agent.run(question)
    
    # 4. PÓS-PROCESSAMENTO (Limpeza da Resposta Final)
    if "Final Answer:" in response:
        response = response.split("Final Answer:", 1)[-1].strip()

    return response


# ======================================================================
# 4. LAYOUT E EXECUÇÃO DO STREAMLIT
# ======================================================================

st.set_page_config(
    page_title="Agente Inteligente de EDA para Fraudes",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Agente de Análise Exploratória de Dados (EDA)")
st.markdown("Este agente é especializado em **Análise Exploratória de Dados (EDA)** de qualquer arquivo CSV.")

# --- Inicialização do Estado da Sessão ---
if 'df' not in st.session_state:
    st.session_state['df'] = None
    st.session_state['memoria_conclusoes'] = ""
    st.session_state['delimiter'] = ',' # Delimitador padrão

# --- Sidebar e Interação do Usuário ---

with st.sidebar:

    st.header("Configuração da LLM")
    api_key = os.getenv("OPENAI_API_KEY", "") 
    openai_api_key = st.text_input("Sua Chave OpenAI API", value='', type="password")

    # 1. Definição da instância LLM
    if openai_api_key:
        llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.0, model_name="gpt-3.5-turbo-16k") 
    else:
        llm = ChatOpenAI(openai_api_key=api_key, temperature=0.0, model_name="gpt-3.5-turbo-16k")
        
    # 2. Lógica de Carregamento de Dados
    st.header("Upload do Dataset")

    # Input do Delimitador
    delimiter_input = st.text_input(
        "Caractere Separador de Colunas (Delimitador)", 
        value=st.session_state['delimiter'], 
        max_chars=1
    )

    if delimiter_input:
        st.session_state['delimiter'] = delimiter_input

    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    
    if uploaded_file is not None and st.session_state['df'] is None:
        if llm is not None:
            load_data(uploaded_file, llm)
        else:
            st.warning("Por favor, insira sua chave da OpenAI e carregue o CSV para inicializar o agente.")
    
    # Mensagem de Informação em Destaque
    st.info(
        "ℹ️ **Antes de carregar um novo arquivo .csv, recarrege a página web (F5).**"
    )

    # Mensagem de Aviso em Destaque
    st.warning(
        "⚠️ **Este agente está em fase experimental.** Em caso de falhas de iteração ou resultados absurdos, por favor refaça a pergunta."
    )

# --- Interface Principal de Perguntas e Respostas ---
   
if st.session_state['df'] is not None and llm is not None:
    
    # --- NOVO TRECHO DE CÓDIGO ---
    # 1. Verifica se as perguntas dinâmicas já foram geradas
    if 'example_questions' not in st.session_state:
        # Define um prompt explícito para forçar o agente a seguir a Regra 10
        # O agente irá gerar as 4 perguntas no formato de lista Markdown.
        question_prompt = "Gere 4 exemplos de perguntas de acordo com a Regra 9: 1) Conclusões Sobre a Análise de Dados Inicial, 2) Cálculo Estatístico 1, 3) Cálculo Estatístico 2, 4) Geração de Gráfico."
        
        st.info("Gerando exemplos de perguntas relevantes para o seu arquivo CSV...")
        
        # 2. Chama o agente para gerar a lista de perguntas
        questions_response = create_and_run_agent(
            st.session_state['uploaded_file_object'], 
            question_prompt, 
            llm, 
            st.session_state['memory_instance']
        )
        
        # 3. Armazena a resposta formatada na sessão para uso futuro
        st.session_state['example_questions'] = questions_response

    # 4. Exibe as perguntas dinâmicas armazenadas
    st.subheader("Exemplos de Perguntas:")
    # O agente deve retornar uma lista Markdown formatada, então basta imprimir
    st.markdown(st.session_state['example_questions'])
    # --- FIM DO NOVO TRECHO ---

    user_question = st.text_input("Insira sua pergunta de EDA aqui:", key="user_input")

    if user_question:
        
        # Roda o agente LangChain
        response = create_and_run_agent(
            st.session_state['uploaded_file_object'], 
            user_question, 
            llm, 
            st.session_state['memory_instance']
        )
        
        # Formata a resposta
        st.subheader("Resposta do Agente:")
        st.markdown(response)
        
        # LÓGICA DE EXIBIÇÃO DE GRÁFICO
        if os.path.exists(TEMP_PLOT_PATH):
            st.subheader("Visualização Gerada:")
            st.image(TEMP_PLOT_PATH)
            
            # Remove o arquivo para que a próxima execução não pegue o gráfico antigo
            os.remove(TEMP_PLOT_PATH) 
            
else:
    st.info("Por favor, carregue um arquivo CSV e insira sua chave da API para começar.")
