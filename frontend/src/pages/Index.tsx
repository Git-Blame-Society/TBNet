'use client';

import React, { useState } from 'react';
import DiagnosticForm from '../components/DiagnosticForm';
import ImageUpload from '../components/ImageUpload';
import DiagnosticResults from '../components/DiagnosticResults';
import { Brain, Shield } from 'lucide-react';

interface DiagnosticResult {
  probability: number;
  verdict: 'positive' | 'negative' | 'inconclusive';
  confidence: number;
  riskFactors: string[];
  recommendations: string[];
  processingTime: number;
}

const Index = () => {
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [result, setResult] = useState<DiagnosticResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Symptom labels matching your backend
  const symptomLabels = [
    'Fever for two weeks',
    'Coughing blood',
    'Sputum mixed with blood',
    'Night sweats',
    'Chest pain',
    'Back pain in certain parts',
    'Shortness of breath',
    'Weight loss',
    'Body feels tired',
    'Lumps that appear around the armpits and neck',
    'Cough and phlegm continuously for two weeks to four weeks',
    'Swollen lymph nodes',
    'Loss of appetite'
  ];

  const handleAnalyze = async () => {
    if (!uploadedImage) {
      alert("Please select an image first!");
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    try {
      const startTime = Date.now();

      // Convert symptoms to binary array (same as your dashboard)
      const symptomBinary = symptomLabels.map(label =>
        selectedSymptoms.includes(label) ? 1 : 0
      );

      // Step 1: Upload symptoms
      const symptomsRes = await fetch("http://localhost:8000/upload-symptoms", {
        method: "POST",
        headers: {
          "Accept": "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ result: symptomBinary }),
      });

      if (!symptomsRes.ok) {
        throw new Error(`Symptoms API failed: ${symptomsRes.statusText}`);
      }

      const symptomsData = await symptomsRes.json();
      const symProb = symptomsData.sym_probability;

      // Step 2: Upload image
      const formData = new FormData();
      formData.append("file", uploadedImage);

      const imageRes = await fetch("http://localhost:8000/upload-image", {
        method: "POST",
        body: formData,
      });

      if (!imageRes.ok) {
        throw new Error(`Image API failed: ${imageRes.statusText}`);
      }

      const imageData = await imageRes.json();
      const imageProb = imageData.image_probability;

      // Calculate final probability (same logic as your dashboard)
      const finalProb = symProb * 0.3 + imageProb * 0.7;
      const prediction = finalProb > 0.5 ? 1 : 0;
      const processingTime = (Date.now() - startTime) / 1000;

      // Convert to the expected result format
      const result: DiagnosticResult = {
        probability: Math.round(finalProb * 100),
        verdict: prediction === 1 ? 'positive' : 'negative',
        confidence: Math.round(Math.max(finalProb, 1 - finalProb) * 100),
        riskFactors: selectedSymptoms.slice(0, 3),
        recommendations: prediction === 1 ? [
          'Immediate consultation with a pulmonologist',
          'Complete sputum culture and sensitivity test',
          'Chest CT scan for detailed lung assessment',
          'Start isolation protocols if TB is confirmed',
          'Contact tracing for recent close contacts'
        ] : [
          'Continue regular health monitoring',
          'Follow up if symptoms persist or worsen',
          'Maintain good respiratory hygiene'
        ],
        processingTime: processingTime
      };

      setResult(result);
    } catch (error) {
      console.error('Error calling AI backend:', error);

      // Fallback result on error
      setResult({
        probability: 0,
        verdict: 'inconclusive',
        confidence: 0,
        riskFactors: ['API Error'],
        recommendations: ['Please check your connection and try again'],
        processingTime: 0
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleNewAnalysis = () => {
    setResult(null);
    setSelectedSymptoms([]);
    setUploadedImage(null);
  };

  const handleRemoveImage = () => {
    setUploadedImage(null);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="relative z-10">
        {/* Header */}
        <header className="border-b border-border/50 bg-background/80 backdrop-blur-sm">
          <div className="container mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/20 glow-primary">
                  <Brain className="w-8 h-8 text-primary" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gradient">TB-AI Diagnostics</h1>
                  <p className="text-muted-foreground text-sm">Advanced Tuberculosis Detection System</p>
                </div>
              </div>

              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Shield className="w-4 h-4 text-success" />
                <span>HIPAA Compliant</span>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="container mx-auto px-6 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Input Section */}
            <div className="space-y-8">
              <DiagnosticForm
                selectedSymptoms={selectedSymptoms}
                onSymptomsChange={setSelectedSymptoms}
                uploadedImage={uploadedImage}
                onAnalyze={handleAnalyze}
              />

              <ImageUpload
                onImageUpload={setUploadedImage}
                uploadedImage={uploadedImage}
                onRemoveImage={handleRemoveImage}
              />
            </div>

            {/* Results Section */}
            <div className="lg:sticky lg:top-8 h-fit">
              <DiagnosticResults
                result={result}
                isAnalyzing={isAnalyzing}
                onNewAnalysis={handleNewAnalysis}
              />
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-border/50 bg-background/80 backdrop-blur-sm mt-16">
          <div className="container mx-auto px-6 py-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center md:text-left">
              <div>
                <h3 className="font-semibold text-foreground mb-2">TB-AI Diagnostics</h3>
                <p className="text-sm text-muted-foreground">
                  Revolutionizing tuberculosis detection with artificial intelligence
                </p>
              </div>
              <div>
                <h4 className="font-medium text-foreground mb-2">Important Notice</h4>
                <p className="text-xs text-muted-foreground">
                  This tool is for diagnostic assistance only. Always consult qualified medical professionals for final diagnosis and treatment decisions.
                </p>
              </div>
              <div>
                <h4 className="font-medium text-foreground mb-2">Privacy & Security</h4>
                <p className="text-xs text-muted-foreground">
                  All medical data is processed securely and in compliance with healthcare privacy regulations.
                </p>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default Index;
