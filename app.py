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

#def load_data(uploaded_file):
#    """Carrega o CSV e inicializa o DataFrame na sessão."""
#    try:
#        df = pd.read_csv(uploaded_file)
#        
#        # Realiza uma análise inicial para preencher a memória (requisito 4)
#        st.session_state['df'] = df
#        st.session_state['memoria_conclusoes'] = initial_analysis_and_memory(df)
#        
#        st.success("Arquivo carregado com sucesso! Pronto para perguntar.")
#        
#    except Exception as e:
#        st.error(f"Erro ao carregar o arquivo: {e}")
#
# Onde você salva o DataFrame, salve também o objeto de upload
#def load_data(uploaded_file):
#    """Carrega o CSV e inicializa o DataFrame e o objeto de arquivo na sessão."""
#    try:
#        df = pd.read_csv(uploaded_file)
#        
#        # Salva o DataFrame e o objeto de arquivo (que LangChain precisa)
#        st.session_state['df'] = df
#        st.session_state['uploaded_file_object'] = uploaded_file # << NOVO
#        st.session_state['memoria_conclusoes'] = initial_analysis_and_memory(df)
#        
#        st.success("Arquivo carregado com sucesso! Pronto para perguntar.")
#        
#    except Exception as e:
#        st.error(f"Erro ao carregar o arquivo: {e}")
#
# Seu código, aproximadamente Linha 40

def load_data(uploaded_file):
    """Carrega o CSV, inicializa o DataFrame e reinicia o ponteiro do objeto de arquivo."""
    try:
        # 1. Lê o arquivo. O ponteiro fica no final.
        df = pd.read_csv(uploaded_file)
        
        # 2. **CRUCIAL**: Reinicia o ponteiro de leitura do objeto de arquivo para o início (byte 0).
        # Isso garante que o LangChain possa ler o arquivo a partir do começo.
        uploaded_file.seek(0) # << LINHA CHAVE ADICIONAL
        
        # 3. Salva o DataFrame e o objeto de arquivo na sessão
        st.session_state['df'] = df
        st.session_state['uploaded_file_object'] = uploaded_file
        st.session_state['memoria_conclusoes'] = initial_analysis_and_memory(df)
        
        st.success("Arquivo carregado com sucesso! Pronto para perguntar.")
        
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")


# --- Módulo 2: Geração de Conclusões Iniciais (A Memória) ---

def initial_analysis_and_memory(df: pd.DataFrame) -> str:
    """Realiza análises chaves para preencher a 'memória' do agente."""
    
    # 1. Análise de Desbalanceamento
    total = len(df)
    fraudes = df['Class'].sum()
    percent_fraude = (fraudes / total) * 100
    
    # 2. Correlações Chaves (foco nas variáveis PCA mais fortes)
    # Calcule a correlação de todas as colunas com 'Class'
    correlations = df.corr()['Class'].sort_values(key=abs, ascending=False).drop('Class')
    top_negative = correlations[correlations < 0].head(3)
    
    # 3. Análise da Variável Amount (para mostrar variabilidade)
    amount_stats = df['Amount'].describe().to_string()
    
    # Constrói o texto de "memória" que o agente usará na Pergunta 4
    memoria_text = f"""
    O agente realizou uma análise inicial do conjunto de dados e obteve as seguintes conclusões:
    
    1. **Desbalanceamento de Classe**: O conjunto de dados é extremamente desbalanceado. A classe positiva (fraude) representa apenas **{percent_fraude:.3f}%** das transações ({fraudes} fraudes em {total} transações). Qualquer modelo de classificação deve usar métricas como AUPRC.
    
    2. **Variáveis PCA Relevantes**: As variáveis resultantes da PCA que demonstram a maior correlação com a fraude ('Class') são:
        * **V17** (negativa, indicando que valores baixos desta componente estão ligados à fraude).
        * **V14** (negativa, similar à V17).
        * **V10** (negativa).
        
    3. **Estatísticas da Variável Amount**: O valor das transações ('Amount') possui uma alta variabilidade. As estatísticas descritivas são:\n{amount_stats}
    
    4. **Padrão Temporal (Potencial)**: A variável 'Time' deve ser analisada cuidadosamente para identificar se as fraudes se concentram em períodos específicos do dia (visto que o 'Time' está em segundos desde a primeira transação).
    """
    
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
    st.header("Upload do Dataset")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if uploaded_file is not None and st.session_state['df'] is None:
        load_data(uploaded_file)
        
    st.header("Configuração da LLM")
    
    # Use a chave da .env, mas permite que o usuário sobrescreva
    api_key = os.getenv("OPENAI_API_KEY", "") 
    openai_api_key = st.text_input("Sua Chave OpenAI API", type="password", value=api_key)
    
    if openai_api_key:
        llm = OpenAI(openai_api_key=openai_api_key, temperature=0.0)
    else:
        st.warning("Por favor, insira sua chave da OpenAI ou configure o arquivo .env.")
        llm = None
        
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

