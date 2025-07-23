import React from 'react';
import { Link } from 'wouter';
import { Button } from '@/components/ui/button';
import { 
  Zap, 
  Brain, 
  TrendingUp, 
  Shield, 
  Globe, 
  Cloud,
  BarChart3,
  Sparkles
} from 'lucide-react';

export const Home: React.FC = () => {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white">
        <div className="absolute inset-0 bg-grid-white/[0.02] bg-[size:50px_50px]" />
        <div className="relative container mx-auto px-6 py-24">
          <div className="max-w-4xl mx-auto text-center space-y-8">
            <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur px-4 py-2 rounded-full text-sm">
              <Sparkles className="h-4 w-4" />
              <span>AI-Powered Advertising Platform</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight">
              Welcome to{' '}
              <span className="bg-gradient-to-r from-yellow-400 to-orange-400 bg-clip-text text-transparent">
                Pulser
              </span>
            </h1>
            
            <p className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto">
              Transform your advertising with frontier AI. Achieve 50-100% performance improvements through intelligent optimization.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
              <Link href="/ai">
                <Button size="lg" className="bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600">
                  <Brain className="mr-2 h-5 w-5" />
                  Launch AI Dashboard
                </Button>
              </Link>
              <Button size="lg" variant="outline" className="border-white/20 text-white hover:bg-white/10">
                <BarChart3 className="mr-2 h-5 w-5" />
                View Case Studies
              </Button>
            </div>
            
            <div className="grid grid-cols-3 gap-8 pt-12 text-center">
              <div>
                <div className="text-3xl font-bold">+53%</div>
                <div className="text-sm text-gray-400">Average ROAS Improvement</div>
              </div>
              <div>
                <div className="text-3xl font-bold">600%</div>
                <div className="text-sm text-gray-400">Weather Ad Growth</div>
              </div>
              <div>
                <div className="text-3xl font-bold">30%</div>
                <div className="text-sm text-gray-400">Better Attribution</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-24 bg-gray-50">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">
              Frontier AI for Modern Advertising
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Pulser combines cutting-edge AI with proven advertising strategies to deliver unprecedented results.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <FeatureCard
              icon={Brain}
              title="7-Element Prompt Engine"
              description="Microsoft's proven structure for 23% CTR improvement through intelligent prompt optimization."
              link="/ai/prompts"
            />
            <FeatureCard
              icon={Cloud}
              title="Weather-Responsive Ads"
              description="Dynamic campaigns that adapt to weather conditions for up to 600% growth in sales."
              link="/ai/weather"
            />
            <FeatureCard
              icon={TrendingUp}
              title="Unified Attribution"
              description="30% more accurate attribution combining MMM, MTA, and incrementality testing."
              link="/ai/attribution"
            />
            <FeatureCard
              icon={Globe}
              title="Platform Automation"
              description="Seamless integration with TikTok Smart+ and Meta Advantage+ for automated optimization."
              link="/ai/platforms"
            />
            <FeatureCard
              icon={Shield}
              title="Bias Detection"
              description="Ensure fair and ethical advertising across all demographics with AI-powered monitoring."
              link="/ai/bias"
            />
            <FeatureCard
              icon={Zap}
              title="Real-Time Optimization"
              description="Dynamic campaign adjustment using multi-armed bandits and evolutionary algorithms."
              link="/ai/optimization"
            />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-gradient-to-r from-purple-600 to-blue-600 text-white">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-4xl font-bold mb-6">
            Ready to Transform Your Advertising?
          </h2>
          <p className="text-xl mb-8 max-w-2xl mx-auto">
            Join leading brands using Pulser's AI to achieve breakthrough performance.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/ai">
              <Button size="lg" className="bg-white text-purple-600 hover:bg-gray-100">
                Get Started Free
              </Button>
            </Link>
            <Button size="lg" variant="outline" className="border-white text-white hover:bg-white/10">
              Schedule Demo
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
};

interface FeatureCardProps {
  icon: React.ElementType;
  title: string;
  description: string;
  link: string;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ icon: Icon, title, description, link }) => (
  <Link href={link}>
    <div className="bg-white p-6 rounded-xl shadow-sm hover:shadow-lg transition-shadow cursor-pointer group">
      <div className="inline-flex p-3 bg-gradient-to-br from-purple-100 to-blue-100 rounded-lg mb-4 group-hover:scale-110 transition-transform">
        <Icon className="h-6 w-6 text-purple-600" />
      </div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-gray-600 mb-4">{description}</p>
      <span className="text-purple-600 font-medium group-hover:underline">
        Learn more →
      </span>
    </div>
  </Link>
);