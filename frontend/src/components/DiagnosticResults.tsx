'use client';

import React from 'react';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import {
  Brain,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Clock,
  FileText,
  Download
} from 'lucide-react';

interface DiagnosticResult {
  probability: number;
  verdict: 'positive' | 'negative' | 'inconclusive';
  confidence: number;
  riskFactors: string[];
  recommendations: string[];
  processingTime: number;
}

interface DiagnosticResultsProps {
  result: DiagnosticResult | null;
  isAnalyzing: boolean;
  onNewAnalysis: () => void;
}

const DiagnosticResults: React.FC<DiagnosticResultsProps> = ({
  result,
  isAnalyzing,
  onNewAnalysis
}) => {
  if (isAnalyzing) {
    return (
      <Card className="diagnostic-card p-6">
        <div className="flex flex-col items-center justify-center py-12 space-y-4">
          <div className="relative">
            <Brain className="w-12 h-12 text-primary animate-pulse" />
            <div className="absolute inset-0 rounded-full border-2 border-primary/30 border-t-primary animate-spin"></div>
          </div>
          <div className="text-center space-y-2">
            <h3 className="text-xl font-semibold text-foreground">AI Analysis in Progress</h3>
            <p className="text-muted-foreground">
              Processing medical data and imaging...
            </p>
          </div>
        </div>
      </Card>
    );
  }

  if (!result) {
    return (
      <Card className="diagnostic-card p-6">
        <div className="text-center py-12 space-y-4">
          <div className="p-4 rounded-full bg-muted/20 w-fit mx-auto">
            <FileText className="w-8 h-8 text-muted-foreground" />
          </div>
          <div className="space-y-2">
            <h3 className="text-xl font-semibold text-foreground">Awaiting Analysis</h3>
            <p className="text-muted-foreground">
              Complete the symptom assessment and upload medical imaging to begin AI diagnosis
            </p>
          </div>
        </div>
      </Card>
    );
  }

  const getVerdictIcon = () => {
    switch (result.verdict) {
      case 'positive':
        return <XCircle className="w-6 h-6 text-destructive" />;
      case 'negative':
        return <CheckCircle2 className="w-6 h-6 text-success" />;
      case 'inconclusive':
        return <AlertTriangle className="w-6 h-6 text-warning" />;
    }
  };

  const getVerdictColor = () => {
    switch (result.verdict) {
      case 'positive':
        return 'destructive';
      case 'negative':
        return 'default';
      case 'inconclusive':
        return 'secondary';
    }
  };

  const getVerdictText = () => {
    switch (result.verdict) {
      case 'positive':
        return 'TB Positive - Requires Immediate Medical Attention';
      case 'negative':
        return 'TB Negative - Low Risk Detected';
      case 'inconclusive':
        return 'Inconclusive - Additional Testing Recommended';
    }
  };

  return (
    <Card className="diagnostic-card p-6">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/20">
            <Brain className="w-6 h-6 text-primary" />
          </div>
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-foreground">AI Diagnostic Results</h2>
            <div className="flex items-center gap-2 text-muted-foreground">
              <Clock className="w-4 h-4" />
              <span className="text-sm">Analysis completed in {result.processingTime}s</span>
            </div>
          </div>
        </div>

        {/* Main Result */}
        <div className="p-6 rounded-lg border-2 border-border/50 bg-card/30">
          <div className="flex items-center gap-4 mb-4">
            {getVerdictIcon()}
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-foreground">
                {getVerdictText()}
              </h3>
              <Badge variant={getVerdictColor() as any} className="mt-1">
                {result.verdict.toUpperCase()}
              </Badge>
            </div>
          </div>

          {/* Probability and Confidence */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">TB Probability</span>
                <span className="font-medium">{result.probability}%</span>
              </div>
              <Progress
                value={result.probability}
                className="h-2"
              />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">AI Confidence</span>
                <span className="font-medium">{result.confidence}%</span>
              </div>
              <Progress
                value={result.confidence}
                className="h-2"
              />
            </div>
          </div>
        </div>

        {/* Risk Factors */}
        {result.riskFactors.length > 0 && (
          <div className="space-y-3">
            <h4 className="font-semibold text-foreground flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-warning" />
              Identified Risk Factors
            </h4>
            <div className="grid grid-cols-1 gap-2">
              {result.riskFactors.map((factor, index) => (
                <div
                  key={index}
                  className="p-3 rounded-lg bg-warning/10 border border-warning/20 text-sm"
                >
                  {factor}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recommendations */}
        <div className="space-y-3">
          <h4 className="font-semibold text-foreground flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-success" />
            Medical Recommendations
          </h4>
          <div className="space-y-2">
            {result.recommendations.map((rec, index) => (
              <div
                key={index}
                className="flex items-start gap-3 p-3 rounded-lg bg-success/10 border border-success/20"
              >
                <div className="w-2 h-2 rounded-full bg-success mt-2 flex-shrink-0"></div>
                <p className="text-sm text-foreground">{rec}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="flex flex-col sm:flex-row gap-3 pt-4 border-t border-border/50">
          <Button
            onClick={onNewAnalysis}
            variant="outline"
            className="flex-1 border-primary/50 text-primary hover:bg-primary/10"
          >
            Run New Analysis
          </Button>
          <Button className="bg-primary hover:bg-primary/90 text-primary-foreground glow-primary">
            <Download className="w-4 h-4 mr-2" />
            Export Report
          </Button>
        </div>
      </div>
    </Card>
  );
};

export default DiagnosticResults;
