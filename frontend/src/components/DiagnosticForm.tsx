'use client';

import React, { useState } from 'react';
import { Card } from './ui/card';
import { Checkbox } from './ui/checkbox';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Stethoscope, AlertTriangle } from 'lucide-react';

const symptoms = [
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

interface DiagnosticFormProps {
  selectedSymptoms: string[];
  onSymptomsChange: (symptoms: string[]) => void;
  uploadedImage: File | null;
  onAnalyze: () => void;
}

const DiagnosticForm: React.FC<DiagnosticFormProps> = ({
  selectedSymptoms,
  onSymptomsChange,
  uploadedImage,
  onAnalyze
}) => {
  const handleSymptomChange = (symptom: string, checked: boolean) => {
    if (checked) {
      onSymptomsChange([...selectedSymptoms, symptom]);
    } else {
      onSymptomsChange(selectedSymptoms.filter(s => s !== symptom));
    }
  };

  return (
    <Card className="diagnostic-card p-6">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-primary/20">
            <Stethoscope className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-foreground">Symptom Assessment</h2>
            <p className="text-muted-foreground">Select all symptoms that apply to the patient</p>
          </div>
        </div>

        {/* Symptoms Grid */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-foreground">Clinical Symptoms</h3>
            <Badge variant="secondary" className="text-sm">
              {selectedSymptoms.length} selected
            </Badge>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {symptoms.map((symptom, index) => (
              <div
                key={index}
                className="flex items-center space-x-3 p-3 rounded-lg border border-border/50 hover:border-primary/30 transition-all duration-200 hover:bg-card/50"
              >
                <Checkbox
                  id={`symptom-${index}`}
                  checked={selectedSymptoms.includes(symptom)}
                  onCheckedChange={(checked) => handleSymptomChange(symptom, checked as boolean)}
                  className="data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                />
                <label
                  htmlFor={`symptom-${index}`}
                  className="text-sm text-foreground cursor-pointer flex-1 leading-relaxed"
                >
                  {symptom}
                </label>
              </div>
            ))}
          </div>
        </div>

        {/* Analysis Button */}
        <div className="pt-4 border-t border-border/50">
          <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <AlertTriangle className="w-4 h-4" />
              <span>Ensure all symptoms are accurately selected before analysis</span>
            </div>

            <Button
              onClick={onAnalyze}
              disabled={selectedSymptoms.length === 0 || !uploadedImage}
              className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-2 glow-primary disabled:opacity-50 disabled:hover:bg-primary"
            >
              Run AI Analysis
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default DiagnosticForm;
