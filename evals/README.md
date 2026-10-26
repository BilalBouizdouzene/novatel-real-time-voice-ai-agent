# Évaluation audio et robustesse

## Jeu textuel reproductible

Le benchmark normal évalue le routage et les sources sans télécharger de modèle :

```powershell
python -m scripts.evaluate_voice_agent
python -m scripts.evaluate_voice_agent --dataset evals/robustness_cases.jsonl
python -m scripts.evaluate_voice_agent --strategies bm25,dense,hybrid
```

`bm25` est local et léger. `dense` et `hybrid` peuvent télécharger le modèle d'embeddings configuré et écrire un index FAISS sous `data/`; vérifie l'espace disque avant de les lancer. Chaque résultat expose le mode effectivement utilisé (`faiss`, `bm25`, fallback ou hybride), afin qu'un fallback ne soit pas présenté comme une mesure dense. Les scores RRF et CrossEncoder ne sont pas des probabilités. Un reranker s'active explicitement avec `--reranker-model NOM_DU_MODELE`.

## WAV de référence

1. N'utilise que des enregistrements obtenus avec consentement, anonymisés, et autorisés pour l'évaluation.
2. Convertis chaque fichier en WAV mono, PCM signé 16 bits, 16 kHz. N'ajoute pas de nom, téléphone, adresse ou identifiant client dans les noms de fichier.
3. Copie le format de `audio_cases.template.jsonl` vers un fichier local non versionné, remplace le chemin, et fournis une transcription humaine relue comme `reference_transcript`.
4. Le WER/CER est calculé uniquement en comparant cette référence à la sortie réelle de Whisper. Un audio vide ou un mauvais format est rejeté.
5. `--noise-snr-db 20 10` ajoute en mémoire du bruit blanc synthétique à deux rapports signal/bruit; les originaux ne sont pas modifiés. Ce test contrôlé ne remplace pas des enregistrements dans de vrais environnements bruyants.

```powershell
python -m scripts.evaluate_voice_agent `
  --dataset evals/my_private_audio_cases.jsonl `
  --noise-snr-db 20 10 `
  --include-transcripts `
  --output reports/audio-evaluation.json
```

Sans `--include-transcripts`, le rapport ne contient pas les transcriptions brutes. Le chemin JSONL et les identifiants de cas restent visibles. Le dossier de rapports, les WAV privés et le JSONL audio ne doivent pas être commités.

Le rapport sépare les conditions `clean` et bruitées, fournit le WER/CER macro par énoncé et le facteur temps réel (temps de calcul ASR / durée WAV). Il lance aussi le retrieval sur les transcriptions reconnues, ce qui donne une mesure du couplage ASR→RAG. Les résultats reflètent uniquement le matériel, les versions, le modèle et le corpus exécutés; ils ne sont jamais prédéfinis.

## Interruption, ambiguïté et charge

`robustness_cases.jsonl` couvre des variantes de formulation/erreurs ASR plausibles, questions courtes, social, ambiguïté, hors périmètre et injection textuelle. Les tests unitaires valident que les questions non pertinentes s'abstiennent, que les preuves restent des données non fiables et que le barge-in ne mélange pas les tours. Le jeu texte n'évalue pas l'accent acoustique; il faut des WAV annotés pour le mesurer.

Le script de charge WebSocket n'est pas inclus dans la CI, car il nécessite un serveur démarré. Démarre le backend puis lance `python -m scripts.load_test_websocket --concurrency 3`. Sans fichier audio, il mesure seulement les connexions et le transport VAD sur silence. Avec `--audio-wav`, il appelle le vrai pipeline : cela consomme CPU/mémoire et peut contacter le fournisseur LLM configuré et le moteur TTS. Commence par une seule session et des WAV consentis. Le rapport donne les percentiles de latence par composant lorsque de vrais tours sont complétés. Ce test n'est ni un test de pénétration ni une garantie de capacité de production.
