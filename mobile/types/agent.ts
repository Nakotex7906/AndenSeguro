export type AgentStatus   = 'on_duty' | 'off_duty' | 'on_break';
export type ResponseLevel = 'optimal' | 'normal' | 'critical';

export interface ActivityItem {
  id: string;
  title: string;
  description: string;
  timestamp: string;
  type: 'alert' | 'patrol' | 'report';
}

export interface Agent {
  id: string;
  name: string;
  badge: string;
  status: AgentStatus;
  assignment: string;
  photoUri?: string;
  phone?: string;
  email?: string;
  interventions: number;
  interventionsDelta: string;
  avgResponseTime: string;
  responseLevel: ResponseLevel;
  recentActivity: ActivityItem[];
}
