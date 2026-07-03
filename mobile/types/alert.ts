export type AlertStatus = 'in_progress' | 'completed' | 'false_alarm' | 'pending';
export type RiskLevel  = 'high' | 'medium' | 'low';

export interface SuspectProfile {
  ageRange: string;
  clothing: string[];
  height: string;
  sex: string;
}

export interface IncidentAlert {
  id: string;
  status: AlertStatus;
  riskLevel: RiskLevel;
  zone: string;
  suspect: SuspectProfile;
  cameraFrameUri?: string;
  detectedAt: string;
  respondedAt?: string;
}
