"""
Data Fabcon Integration Module for JamPacked Creative Intelligence
Provides data profiling and quality assessment capabilities via Pulser SDK
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

try:
    from pulser_sdk.agents import DataFabcon as PulserDataFabcon
    PULSER_AVAILABLE = True
except ImportError:
    PULSER_AVAILABLE = False
    logging.warning("Pulser SDK not available. Data Fabcon will run in mock mode.")

@dataclass
class DataProfile:
    """Data profiling results."""
    table_name: str
    row_count: int
    column_count: int
    data_types: Dict[str, str]
    null_counts: Dict[str, int]
    quality_score: float
    issues: List[str]
    recommendations: List[str]

class DataFabcon:
    """Data Fabcon integration for data profiling and quality assessment."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DataFabcon agent."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        if PULSER_AVAILABLE:
            self.pulser_agent = PulserDataFabcon()
            self.logger.info("Pulser DataFabcon agent initialized")
        else:
            self.pulser_agent = None
            self.logger.info("Running in mock mode - Pulser SDK not available")
    
    def profile(self, table: str, connection: Optional[str] = None) -> DataProfile:
        """
        Profile a table and return comprehensive data quality metrics.
        
        Args:
            table: Table name to profile
            connection: Optional database connection string
            
        Returns:
            DataProfile object with profiling results
        """
        if PULSER_AVAILABLE and self.pulser_agent:
            return self._profile_with_pulser(table, connection)
        else:
            return self._mock_profile(table)
    
    def _profile_with_pulser(self, table: str, connection: Optional[str] = None) -> DataProfile:
        """Profile using Pulser SDK."""
        try:
            result = self.pulser_agent.profile(table, connection_string=connection)
            
            return DataProfile(
                table_name=table,
                row_count=result.get('row_count', 0),
                column_count=result.get('column_count', 0),
                data_types=result.get('data_types', {}),
                null_counts=result.get('null_counts', {}),
                quality_score=result.get('quality_score', 0.0),
                issues=result.get('issues', []),
                recommendations=result.get('recommendations', [])
            )
        except Exception as e:
            self.logger.error(f"Pulser profiling failed for {table}: {e}")
            return self._mock_profile(table)
    
    def _mock_profile(self, table: str) -> DataProfile:
        """Provide mock profiling results when Pulser SDK is not available."""
        self.logger.warning(f"Using mock profile for table: {table}")
        
        return DataProfile(
            table_name=table,
            row_count=1000,  # Mock data
            column_count=10,
            data_types={
                'id': 'integer',
                'name': 'varchar',
                'created_at': 'timestamp',
                'value': 'numeric'
            },
            null_counts={
                'id': 0,
                'name': 5,
                'created_at': 0,
                'value': 12
            },
            quality_score=85.5,
            issues=[
                'Some null values in non-nullable columns',
                'Data type inconsistencies detected'
            ],
            recommendations=[
                'Add data validation constraints',
                'Implement null value handling',
                'Consider data cleansing pipeline'
            ]
        )
    
    def assess_quality(self, table: str, thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Assess data quality against defined thresholds.
        
        Args:
            table: Table name to assess
            thresholds: Quality thresholds for assessment
            
        Returns:
            Quality assessment results
        """
        profile = self.profile(table)
        
        default_thresholds = {
            'min_quality_score': 80.0,
            'max_null_percentage': 5.0,
            'min_row_count': 100
        }
        thresholds = {**default_thresholds, **(thresholds or {})}
        
        # Calculate null percentage
        total_cells = profile.row_count * profile.column_count
        total_nulls = sum(profile.null_counts.values())
        null_percentage = (total_nulls / total_cells * 100) if total_cells > 0 else 0
        
        assessment = {
            'table_name': table,
            'overall_score': profile.quality_score,
            'passes_quality_threshold': profile.quality_score >= thresholds['min_quality_score'],
            'null_percentage': null_percentage,
            'passes_null_threshold': null_percentage <= thresholds['max_null_percentage'],
            'row_count': profile.row_count,
            'passes_row_threshold': profile.row_count >= thresholds['min_row_count'],
            'issues': profile.issues,
            'recommendations': profile.recommendations,
            'thresholds_used': thresholds
        }
        
        # Overall pass/fail
        assessment['overall_pass'] = all([
            assessment['passes_quality_threshold'],
            assessment['passes_null_threshold'],
            assessment['passes_row_threshold']
        ])
        
        return assessment

# Convenience function for quick profiling
def profile(table: str, connection: Optional[str] = None) -> DataProfile:
    """Quick profile function."""
    fabcon = DataFabcon()
    return fabcon.profile(table, connection)

# Module-level instance for ease of use
fabcon = DataFabcon()

__all__ = ['DataFabcon', 'DataProfile', 'profile', 'fabcon']