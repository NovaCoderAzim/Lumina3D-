import React, { Component, type ReactNode } from 'react';

interface Props {
  fallback?: (error: Error, reset: () => void) => ReactNode;
  onError?: (error: Error) => void;
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ModelErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ModelErrorBoundary caught error during 3D model loading:', error, errorInfo);
    this.props.onError?.(error);
  }

  reset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError && this.state.error) {
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.reset);
      }
      return null;
    }
    return this.props.children;
  }
}
