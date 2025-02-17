"""Module containing algorithms compatible with niapy framework modified for use in the MSA"""

from tools.algorithms.fa import FireflyAlgorithm
from tools.algorithms.pso import (
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
