import streamlit as st
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from dotenv import load_dotenv

# Carrega a variável de ambiente (Chave da API)
load_dotenv()

# Inicialização no escopo global para evitar NameError
llm = None 

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

def load_data(uploaded_file, llm_instance): # << Adicione 'llm_instance' aqui
    """Carrega o CSV, inicializa o DataFrame e reinicia o ponteiro do objeto de arquivo."""
    try:
        # 1. Lê o arquivo.
        df = pd.read_csv(uploaded_file)
        
        # 2. Reinicia o ponteiro de leitura do objeto de arquivo.
        uploaded_file.seek(0)
        
        # 3. Chama a função de memória com o LLM para análise autônoma
        # Note que você passa 'df' e 'llm_instance' (o novo nome do parâmetro)
        st.session_state['memoria_conclusoes'] = initial_analysis_and_memory(df, llm_instance)
        
        # 4. Salva o DataFrame e o objeto de arquivo na sessão
        st.session_state['df'] = df
        st.session_state['uploaded_file_object'] = uploaded_file

        st.success("Arquivo carregado e análise inicial autônoma concluída! Pronto para perguntar.")
        
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo ou gerar análise inicial: {e}")


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
        prefix="Você deve executar uma Análise Exploratória de Dados (EDA) e retornar um resumo em texto. Não use o formato Thought/Action. Liste as colunas numéricas, estatísticas chave, e quaisquer colunas que pareçam desbalanceadas ou com alta correlação.",
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    try:
        # Pede ao LLM para fazer a análise de 5 minutos sobre o novo CSV
        return temp_agent.run(prompt)
    except Exception as e:
        return f"Falha ao gerar análise inicial autônoma: {e}"


# Analise de conclusões fixas para o arquivo CREDITCARD.CSV
#
#def initial_analysis_and_memory(df: pd.DataFrame) -> str:
#    """Realiza análises chaves para preencher a 'memória' do agente."""
#    
#    # 1. Análise de Desbalanceamento
#    total = len(df)
#    fraudes = df['Class'].sum()
#    percent_fraude = (fraudes / total) * 100
#    
#    # 2. Correlações Chaves (foco nas variáveis PCA mais fortes)
#    # Calcule a correlação de todas as colunas com 'Class'
#    correlations = df.corr()['Class'].sort_values(key=abs, ascending=False).drop('Class')
#    top_negative = correlations[correlations < 0].head(3)
#    
#    # 3. Análise da Variável Amount (para mostrar variabilidade)
#    amount_stats = df['Amount'].describe().to_string()
#    
#    # Constrói o texto de "memória" que o agente usará na Pergunta 4
#    memoria_text = f"""
#    O agente realizou uma análise inicial do conjunto de dados e obteve as seguintes conclusões:
#    
#    1. **Desbalanceamento de Classe**: O conjunto de dados é extremamente desbalanceado. A classe positiva (fraude) representa apenas **{percent_fraude:.3f}%** das transações ({fraudes} fraudes em {total} transações). Qualquer modelo de classificação deve usar métricas como AUPRC.
#    
#    2. **Variáveis PCA Relevantes**: As variáveis resultantes da PCA que demonstram a maior correlação com a fraude ('Class') são:
#        * **V17** (negativa, indicando que valores baixos desta componente estão ligados à fraude).
#        * **V14** (negativa, similar à V17).
#        * **V10** (negativa).
#        
#    3. **Estatísticas da Variável Amount**: O valor das transações ('Amount') possui uma alta variabilidade. As estatísticas descritivas são:\n{amount_stats}
#    
#    4. **Padrão Temporal (Potencial)**: A variável 'Time' deve ser analisada cuidadosamente para identificar se as fraudes se concentram em períodos específicos do dia (visto que o 'Time' está em segundos desde a primeira transação).
#    """
#    
#    return memoria_text


# Nova função para gerar conclusões com analise da LLM, flexibilizando o código para aceitar qualquer CSV

def initial_analysis_and_memory(df: pd.DataFrame, llm_instance) -> str:
    """Gera o texto de memória de forma autônoma usando o LLM."""
    
    # 1. Prompt genérico para análise inicial:
    prompt = f"""
    Realize uma análise exploratória (EDA) detalhada deste conjunto de dados.
    Suas conclusões devem cobrir:
    1.  **Tipos de Dados**: Quais colunas são numéricas, e quais são categóricas/binárias?
    2.  **Distribuições**: Quais variáveis têm a maior variância ou a distribuição mais assimétrica?
    3.  **Outliers/Desbalanceamento**: Existe alguma coluna binária que está altamente desbalanceada (como 'Class' no dataset original de fraudes) ou colunas com outliers extremos (como 'Amount')?
    4.  **Correlações**: Quais são as duas correlações mais fortes entre quaisquer duas colunas (incluindo a correlação com a coluna alvo, se ela for clara)?
    """
    
    st.info("Gerando análise inicial autônoma do novo CSV. Isso pode levar alguns segundos...")
    
    # 2. Executa o Agente LLM para obter o resumo:
    memoria_text = run_llm_analysis(df, prompt, llm_instance)
    
    return memoria_text


# --- Módulo 3: Criação e Execução do Agente LangChain ---

def create_and_run_agent(file_input, question, llm):
    """Cria e executa o agente LangChain para gerar código Python e responder."""
    # 1. CRUCIAL: Reinicia o ponteiro do arquivo para o início.
    # Isso resolve o EmptyDataError para a segunda (e subsequentes) chamadas.
    try:
        file_input.seek(0)
    except Exception as e:
        # Adicione um tratamento simples, caso o objeto não tenha o método .seek()
        # (Embora objetos de upload do Streamlit geralmente tenham)
        st.warning(f"Não foi possível reiniciar o ponteiro do arquivo: {e}")

    # 2. Cria o agente, que lerá o arquivo do início
    
    # Define as configurações do agente
    agent = create_csv_agent(
        llm,
        file_input, 
        verbose=True, 
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True, 
        agent_executor_kwargs={"handle_parsing_errors": True},
        #prefix="Você é um especialista em análise de dados e estatística. Seu objetivo é ajudar um usuário a realizar Exploratory Data Analysis (EDA) em um DataFrame pandas chamado 'df', que contém dados de transações de cartão de crédito. As colunas V1 a V28 são o resultado de uma transformação PCA. A coluna 'Class' (1=fraude, 0=normal) é o alvo. Sempre que possível e aplicável, **gere um gráfico** para visualizar o resultado, salvando-o em um arquivo .png e exibindo o arquivo. Não use a função `print()`, apenas gere o resultado final.",
        prefix="Você é um especialista em análise de dados e estatística. Seu objetivo não é apenas gerar código Python e gráficos, mas **sempre interpretar a saída ou o gráfico gerado** antes de retornar a resposta. A resposta final deve ser a **conclusão analítica** em texto, seguida pelo artefato (o gráfico). Analise as correlações, distribuições e padrões e explique o que encontrou.",
        max_iterations=15 # Limita o número de tentativas
    )

    try:
        # Executa o agente
        with st.spinner("O Agente está pensando, gerando e executando o código Python..."):
            response = agent.run(question)
            return response
            
    except Exception as e:
        # Se o agente falhar (ex: erro no código Python gerado)
        return f"O agente encontrou um erro: {e}. Por favor, tente reformular a pergunta. Detalhes: A execução do código falhou."


# --- Sidebar e Interação do Usuário ---

with st.sidebar:
    #Este bloco abaixo estava mantendo o streamlit sempre esperando a chave api da openai
    #st.header("Upload do Dataset")
    #uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    ##Chamada anterior, somente para CREDITCARD.CSV
    ##if uploaded_file is not None and st.session_state['df'] is None:
    ##    load_data(uploaded_file) 
    # 
    ## Novo bloco de código, para permitir tratar via LLM qualquer CSV
    #if uploaded_file is not None and st.session_state['df'] is None:
    #    # Mude a chamada para garantir que o LLM esteja disponível
    #    if llm is not None:
    #        load_data(uploaded_file, llm) # Passe o LLM como argumento
    #    else:
    #        st.warning("Carregue o CSV e insira a chave API para inicializar o agente.")

    #st.header("Configuração da LLM")
    #
    ## Use a chave da .env, mas permite que o usuário sobrescreva
    #api_key = os.getenv("OPENAI_API_KEY", "") 
    #openai_api_key = st.text_input("Sua Chave OpenAI API", type="password", value=api_key)
    #
    #if openai_api_key:
    #    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.0)
    #else:
    #    st.warning("Por favor, insira sua chave da OpenAI ou configure o arquivo .env.")
    #    llm = None

    #Correção do bloco acima
    st.header("Upload do Dataset")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    
    st.header("Configuração da LLM")
    api_key = os.getenv("OPENAI_API_KEY", "") 
    openai_api_key = st.text_input("Sua Chave OpenAI API", type="password", value=api_key)
    
    # 1. Definição da instância LLM
    if openai_api_key:
        # Nota: Você pode usar a otimização com st.cache_resource aqui!
        llm = OpenAI(openai_api_key=openai_api_key, temperature=0.0) 
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

