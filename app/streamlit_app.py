import streamlit as st
import pandas as pd
import time
import sys
import os

# Permite a importação da pasta src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_predict import predict_fit
from src.recommendation import recomendar_vagas
from src.skills import comparar_skills
MODULOS_CARREGADOS = True

st.set_page_config(page_title="JobMatch AI", page_icon="🎯", layout="centered")
st.title("JobMatch AI 🎯")
st.markdown("Recomendação Inteligente de Vagas e Análise de Compatibilidade.")
st.divider()

if not MODULOS_CARREGADOS:
    st.warning("⚠️ Módulos do backend (`src/`) não encontrados. Rodando com dados simulados conforme Issue #4.")

st.subheader("1. Insira seu Currículo")
curriculo_usuario = st.text_area(
    label="Cole aqui o seu perfil profissional ou currículo completo:",
    height=200,
    placeholder="Ex: Estudante de Engenharia Elétrica com experiência em Python..."
)

if st.button("Executar Análise de Compatibilidade", type="primary"):
    if len(curriculo_usuario.strip()) < 30:
        st.error("O currículo inserido é muito curto. Por favor, detalhe mais suas experiências.")
    else:
        with st.spinner("Analisando perfil contra a base de vagas e extraindo skills..."):
            
            if MODULOS_CARREGADOS:
                df_vagas = pd.read_csv('data/sample/vagas_exemplo.csv')
                top_vagas = recomendar_vagas(curriculo_usuario, df_vagas)
                
                vaga_principal = top_vagas.iloc[0]
                id_da_vaga = vaga_principal['job_id']

                # 1. Vai na base original e "pesca" o texto completo usando o ID da vaga
                texto_completo = df_vagas[df_vagas['job_id'] == id_da_vaga]['texto_vaga_completo'].iloc[0]

                # 2. Envia o texto completo recuperado para os modelos
                resultado_modelo = predict_fit(curriculo_usuario, texto_completo)
                score = resultado_modelo['score_confianca']
                classificacao = resultado_modelo['status'] 

                skills = comparar_skills(curriculo_usuario, texto_completo)
                skills_comp = skills['skills_compativeis']
                skills_falt = skills['skills_faltantes']
                
            else:
                time.sleep(2) 
                score = 0.88
                classificacao = "Fit"
                skills_comp = ["Python", "Machine Learning", "Pesquisa Científica"]
                skills_falt = ["Docker", "Deploy em Nuvem"]
                top_vagas_mock = [
                    "Cientista de Dados Júnior - Tech Corp (Score: 88%)",
                    "Analista de Machine Learning - SiDi (Score: 85%)",
                    "Pesquisador de IA - Inova Ltda (Score: 79%)",
                    "Desenvolvedor Python - DevTech (Score: 76%)",
                    "Engenheiro de Dados - FinanceX (Score: 72%)"
                ]

        st.success("✅ Análise concluída!")
        st.divider()
        
        st.subheader("2. Resultados da Análise")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Score de Aderência (Top 1)", value=f"{score * 100 if score <= 1 else score:.0f}%")
        with col2:
            st.metric(label="Classificação", value=f"{classificacao} {'✅' if classificacao == 'Fit' else '❌'}")
            
        with st.expander("🛠️ Análise de Skills (Vaga Principal)"):
            c_comp, c_falt = st.columns(2)
            with c_comp:
                st.markdown("**🎯 Skills Compatíveis:**")
                for s in skills_comp: st.markdown(f"- {s}")
            with c_falt:
                st.markdown("**🚧 Skills Faltantes:**")
                for s in skills_falt: st.markdown(f"- {s}")
                
        with st.expander("🏆 Ranking Top-5 Vagas Recomendadas"):
            if MODULOS_CARREGADOS:
                for idx, row in top_vagas.iterrows():
                    st.write(f"**{idx + 1}. {row['title']}** na {row['company_name']}")
            else:
                for vaga in top_vagas_mock:
                    st.write(f"- {vaga}")
                    
        with st.expander("📈 Sugestão de Desenvolvimento"):
            st.info("Recomendamos focar nas habilidades listadas como 'Faltantes' para aumentar sua aderência às oportunidades ideais para o seu perfil.")