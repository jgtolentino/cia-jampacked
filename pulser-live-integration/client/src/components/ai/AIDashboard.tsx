import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { 
  TrendingUp, 
  Brain, 
  Cloud, 
  BarChart3, 
  Target,
  Zap,
  Shield,
  Globe
} from 'lucide-react';

interface CampaignMetrics {
  total_campaigns: number;
  active_optimizations: number;
  performance_improvement: number;
  cost_savings: number;
}

interface AIInsight {
  id: string;
  type: 'optimization' | 'warning' | 'opportunity';
  title: string;
  description: string;
  impact: string;
  action: string;
}

export const AIDashboard: React.FC = () => {
  const [selectedTimeframe, setSelectedTimeframe] = useState('7d');
  
  // Fetch dashboard metrics
  const { data: metrics, isLoading: metricsLoading } = useQuery<CampaignMetrics>({
    queryKey: ['ai-metrics', selectedTimeframe],
    queryFn: async () => {
      const response = await fetch(`/api/ai/metrics?timeframe=${selectedTimeframe}`);
      return response.json();
    }
  });

  // Fetch AI insights
  const { data: insights, isLoading: insightsLoading } = useQuery<AIInsight[]>({
    queryKey: ['ai-insights'],
    queryFn: async () => {
      const response = await fetch('/api/ai/insights');
      return response.json();
    }
  });

  const MetricCard = ({ 
    icon: Icon, 
    title, 
    value, 
    change, 
    color 
  }: { 
    icon: any; 
    title: string; 
    value: string; 
    change?: string; 
    color: string;
  }) => (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className={`h-4 w-4 text-${color}-600`} />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {change && (
          <p className="text-xs text-muted-foreground mt-1">
            <TrendingUp className="inline h-3 w-3 mr-1" />
            {change} from last period
          </p>
        )}
      </CardContent>
    </Card>
  );

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">AI Command Center</h1>
          <p className="text-muted-foreground">
            Real-time optimization powered by frontier AI
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">Export Report</Button>
          <Button size="sm">
            <Zap className="h-4 w-4 mr-2" />
            Quick Optimize
          </Button>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          icon={Target}
          title="Active Campaigns"
          value={metrics?.total_campaigns.toString() || '0'}
          change="+12%"
          color="blue"
        />
        <MetricCard
          icon={Brain}
          title="AI Optimizations"
          value={metrics?.active_optimizations.toString() || '0'}
          change="+28%"
          color="purple"
        />
        <MetricCard
          icon={TrendingUp}
          title="Performance Lift"
          value={`+${metrics?.performance_improvement || 0}%`}
          change="+15%"
          color="green"
        />
        <MetricCard
          icon={BarChart3}
          title="Cost Savings"
          value={`$${metrics?.cost_savings?.toLocaleString() || '0'}`}
          change="+32%"
          color="yellow"
        />
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="prompts">Prompt Builder</TabsTrigger>
          <TabsTrigger value="attribution">Attribution</TabsTrigger>
          <TabsTrigger value="weather">Weather Ads</TabsTrigger>
          <TabsTrigger value="platforms">Platforms</TabsTrigger>
          <TabsTrigger value="intelligence">Intelligence</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          {/* AI Insights */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                AI-Powered Insights
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {insights?.map((insight) => (
                  <div
                    key={insight.id}
                    className={`p-4 rounded-lg border ${
                      insight.type === 'warning' 
                        ? 'border-yellow-200 bg-yellow-50' 
                        : insight.type === 'opportunity'
                        ? 'border-green-200 bg-green-50'
                        : 'border-blue-200 bg-blue-50'
                    }`}
                  >
                    <div className="flex justify-between items-start">
                      <div className="space-y-1">
                        <h4 className="font-semibold">{insight.title}</h4>
                        <p className="text-sm text-muted-foreground">
                          {insight.description}
                        </p>
                        <p className="text-sm font-medium">
                          Impact: {insight.impact}
                        </p>
                      </div>
                      <Button size="sm" variant="outline">
                        {insight.action}
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Performance Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Performance Optimization Trend</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-center justify-center text-muted-foreground">
                {/* Chart component would go here */}
                Performance chart visualization
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="prompts">
          <Card>
            <CardHeader>
              <CardTitle>7-Element Prompt Builder</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Create high-performing prompts using Microsoft's proven structure
                </p>
                {/* Prompt builder component would go here */}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="attribution">
          <Card>
            <CardHeader>
              <CardTitle>Unified Attribution Model</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  30% more accurate attribution combining MMM, MTA, and incrementality
                </p>
                {/* Attribution component would go here */}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="weather">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cloud className="h-5 w-5" />
                Weather-Responsive Campaigns
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  600% growth potential with weather-triggered advertising
                </p>
                {/* Weather ads component would go here */}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="platforms">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Globe className="h-5 w-5" />
                Platform Integrations
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="p-4 border rounded-lg">
                  <h4 className="font-semibold mb-2">TikTok Smart+</h4>
                  <p className="text-sm text-muted-foreground">
                    53% ROAS improvement with AI optimization
                  </p>
                  <Button size="sm" className="mt-3">Configure</Button>
                </div>
                <div className="p-4 border rounded-lg">
                  <h4 className="font-semibold mb-2">Meta Advantage+</h4>
                  <p className="text-sm text-muted-foreground">
                    $20B annual run-rate platform integration
                  </p>
                  <Button size="sm" className="mt-3">Configure</Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="intelligence">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Competitive Intelligence
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Real-time market monitoring and bias detection
                </p>
                {/* Intelligence component would go here */}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};