import json
import numpy as np
import time

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    print("--- SNIKTITT PÅ PRODUKT-GJENKJENNING (STEG 2) ---")
    print("Laster inn Feature Bank (hjernen vår med 356 produkter)...\n")
    
    # Laster "hjernen"
    with open("/Users/blekkulf/workspace/norgesgruppen-data/feature_bank.json", "r") as f:
        feature_bank = json.load(f)
        
    print(f"✅ Vektor-database lastet med {len(feature_bank)} unike EAN/Produkt-koder.")
    
    # Simulere at YOLO har klippet ut et produkt fra et butikk-bilde, og at ResNet
    # akkurat har sendt ut en 2048-dimensjonal vektor av dette ukjente klippet.
    # Vi henter ut en tilfeldig vektor fra databasen og legger til litt "støy" for å simulere et dårlig/uskarpt bilde fra hylla
    
    test_key = list(feature_bank.keys())[42]  # Plukker ut produkt nummer 42 tilfeldig
    true_embedding = np.array(feature_bank[test_key]["embedding"])
    
    print("\n[Simulerer YOLO bilde-utklipp]")
    print(f"📸 YOLO fant en boks. Vi klipper det ut fra bildet og kjører gjennom ResNet...")
    time.sleep(1)
    
    # Legger til 30% støy for å simulere dårlig lys, blur, og rot på butikkhylla
    noise = np.random.normal(0, 0.3, true_embedding.shape)
    unknown_embedding = true_embedding + noise
    unknown_embedding = unknown_embedding / np.linalg.norm(unknown_embedding)  # Re-normaliser
    
    print(f"🧠 ResNet genererte en 2048-dimensjonal vektor for det ukjente produktet.")
    print("🔍 Søker i Feature Banken (K-Nearest Neighbors med Cosine Similarity)...\n")
    time.sleep(1.5)
    
    # Kjører Vector Search! 
    results = []
    for ean, data in feature_bank.items():
        db_vector = np.array(data["embedding"])
        sim = cosine_similarity(unknown_embedding, db_vector)
        results.append((ean, data["category_id"], sim))
        
    # Sorter etter mest lik
    results.sort(key=lambda x: x[2], reverse=True)
    
    print("🏆 Topp 3 treff i databasen:")
    print("-" * 50)
    for i in range(3):
        ean, cat_id, sim = results[i]
        if i == 0:
            print(f"🥇 Match 1: EAN {ean:<15} | Kategori ID: {cat_id:<4} | Likhet: {sim*100:.1f}%")
        else:
            print(f"   Match {i+1}: EAN {ean:<15} | Kategori ID: {cat_id:<4} | Likhet: {sim*100:.1f}%")
            
    print("-" * 50)
    print(f"\n💡 Konklusjon: Systemet er {results[0][2]*100:.1f}% sikker på at dette er produkt '{results[0][0]}' (Kategori {results[0][1]}).")
    print(f"Skriptet vil dermed sette \"category_id\": {results[0][1]} på denne YOLO-boksen i submission.json.")

if __name__ == "__main__":
    main()
