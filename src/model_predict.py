import os
import joblib
from src.data_loader import garantir_pastas_projeto

def predict_fit(curriculo: str, descricao_vaga: str) -> dict:
    """
    Recebe um currículo e uma descrição de vaga separados,
    formata no padrão do dataset de treino (Hugging Face) e
    retorna a classificação Fit/No Fit com score de confiança.
    """
    # 1. Recupera os caminhos dinâmicos das pastas do projeto
    pastas = garantir_pastas_projeto()
    model_path = pastas["models"] / "logistic_regression_baseline.pkl"
    vectorizer_path = pastas["models"] / "tfidf_vectorizer.pkl"
    
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Artefatos do modelo não encontrados em {pastas['models']}. Execute o treino primeiro."
        )
    
    # 2. Carrega os artefatos locais
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # 3. Simula o padrão exato de texto que o modelo aprendeu no Hugging Face
    texto_combinado = f"For the given job description << {descricao_vaga} >> {curriculo}"
    
    # 4. Transforma o texto usando o mesmo vetorizador do treino
    X_tfidf = vectorizer.transform([texto_combinado])
    
    # 5. Realiza a inferência
    predicao = model.predict(X_tfidf)[0]
    probabilidades = model.predict_proba(X_tfidf)[0]  # [prob_0, prob_1]
    
    # 6. Mapeia os resultados para o formato de negócio
    label = "Fit" if predicao == 1 else "No Fit"
    confianca = probabilidades[1] if predicao == 1 else probabilidades[0]
    
    return {
        "status": label,
        "score_confianca": round(float(confianca), 4)
    }

if __name__ == "__main__":
    # Teste rápido de fumaça (Sanity Check)
    vaga_teste = "Looking for a Data Scientist skilled in Python, SQL and Machine Learning."
    curriculo_teste = "Experienced Data Scientist with background in Python development and Scikit-Learn."
    
    print("\n=== TESTANDO INFERÊNCIA LOCAL ===")
    try:
        resultado = predict_fit(curriculo_teste, vaga_teste)
        print("Resultado do Teste:", resultado)
    except Exception as e:
        print("Erro ao rodar o teste:", e)