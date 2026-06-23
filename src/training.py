import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Importa as funções do data_loader que você acabou de atualizar
from src.data_loader import carregar_resume_jd_match, garantir_pastas_projeto

def train_baseline_model():
    print("--- INICIANDO PIPELINE DE TREINAMENTO ---")
    
    # 1. Carrega o dado via data_loader (baixa da API se não existir localmente)
    df = carregar_resume_jd_match()
    
    # 2. Binarizar a label conforme critérios de aceite (No Fit = 0, Potential/Good = 1)
    mapeamento = {'No Fit': 0, 'Potential Fit': 1, 'Good Fit': 1}
    df['label_binario'] = df['label'].map(mapeamento)
    
    # 3. Separar Treino e Teste mantendo os mesmos splits do Hugging Face originais
    df_train = df[df['split'] == 'train']
    df_test = df[df['split'] == 'test']
    
    X_train, y_train = df_train['text'], df_train['label_binario']
    X_test, y_test = df_test['text'], df_test['label_binario']
    
    print(f"[INFO] Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")
    
    # 4. Vetorização (TF-IDF Baseline)
    print("[INFO] Extraindo features com TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 5. Treinamento
    print("[INFO] Treinando Regressão Logística...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # 6. Avaliação (Critérios de Aceite batidos)
    y_pred = model.predict(X_test_tfidf)
    print("\n=== METRICAS DO MODELO BASELINE ===")
    print("Acurácia:", round(accuracy_score(y_test, y_pred), 4))
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, target_names=['No Fit', 'Fit']))
    
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    
    # 7. Salvar os modelos usando os caminhos dinâmicos do projeto
    pastas = garantir_pastas_projeto()
    model_path = pastas["models"] / "logistic_regression_baseline.pkl"
    vectorizer_path = pastas["models"] / "tfidf_vectorizer.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\n[SUCESSO] Modelo salvo em: {model_path}")
    print(f"[SUCESSO] Vetorizador salvo em: {vectorizer_path}")

if __name__ == "__main__":
    train_baseline_model()