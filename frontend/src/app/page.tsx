"use client";

import { useState } from "react";
import Navbar from './components/Navbar';
import DataUpload from "./components/DataUpload";
import PredictionResult from "./components/PredictionResult";

export default function Page() {
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnalyze = async () => {
    if (!uploadedImage) {
      alert("Please select an image first!");
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    try {
      // Convert symptoms to binary array
      const symptomLabels = [
        "Fever for two weeks",
        "Coughing blood",
        "Sputum mixed with blood",
        "Night sweats",
        "Chest pain",
        "Back pain in certain parts",
        "Shortness of breath",
        "Weight loss",
        "Body feels tired",
        "Lumps that appear around the armpits and neck",
        "Cough and phlegm continuously for two weeks to four weeks",
        "Swollen lymph nodes",
        "Loss of appetite",
      ];
      const symptomBinary = symptomLabels.map((label) =>
        selectedSymptoms.includes(label) ? 1 : 0
      );

      // Upload symptoms
      const symptomsRes = await fetch("http://localhost:8000/upload-symptoms", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ result: symptomBinary }),
      });
      const symptomsData = await symptomsRes.json();
      const symProb = symptomsData.sym_probability;

      // Upload image
      const formData = new FormData();
      formData.append("file", uploadedImage);
      const imageRes = await fetch("http://localhost:8000/upload-image", {
        method: "POST",
        body: formData,
      });
      const imageData = await imageRes.json();
      const imageProb = imageData.image_probability;

      // Final probability
      const finalProb = symProb * 0.3 + imageProb * 0.7;
      const prediction = finalProb > 0.5 ? "positive" : "negative";

      setResult({
        probability: Math.round(finalProb * 100),
        verdict: prediction,
        confidence: Math.round(Math.max(finalProb, 1 - finalProb) * 100),
        riskFactors: selectedSymptoms.slice(0, 3),
      });
    } catch (err) {
      console.error(err);
      setResult({
        verdict: "inconclusive",
        probability: 0,
        confidence: 0,
        riskFactors: ["Error contacting API"],
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <DataUpload
            selectedSymptoms={selectedSymptoms}
            setSelectedSymptoms={setSelectedSymptoms}
            uploadedImage={uploadedImage}
            setUploadedImage={setUploadedImage}
            onAnalyze={handleAnalyze}
          />

          <PredictionResult result={result} isAnalyzing={isAnalyzing} />
        </div>
      </main>
    </div>
  );
}
