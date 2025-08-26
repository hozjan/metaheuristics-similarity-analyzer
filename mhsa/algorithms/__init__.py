"""Module containing algorithms compatible with niapy framework modified for use in the MHSA"""

from mhsa.algorithms.fa import FireflyAlgorithm
from mhsa.algorithms.pso import (
    ParticleSwarmAlgorithm,
    ParticleSwarmOptimization,
    CenterParticleSwarmOptimization,
    MutatedParticleSwarmOptimization,
    MutatedCenterParticleSwarmOptimization,
    ComprehensiveLearningParticleSwarmOptimizer,
    MutatedCenterUnifiedParticleSwarmOptimization,
    OppositionVelocityClampingParticleSwarmOptimization,
)

__all__ = [
    "FireflyAlgorithm",
    "ParticleSwarmAlgorithm",
    "ParticleSwarmOptimization",
    "CenterParticleSwarmOptimization",
    "MutatedParticleSwarmOptimization",
    "MutatedCenterParticleSwarmOptimization",
    "ComprehensiveLearningParticleSwarmOptimizer",
    "MutatedCenterUnifiedParticleSwarmOptimization",
    "OppositionVelocityClampingParticleSwarmOptimization",
]
