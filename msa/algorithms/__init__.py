"""Module containing algorithms compatible with niapy framework modified for use in the MSA"""

from msa.algorithms.fa import FireflyAlgorithm
from msa.algorithms.pso import (
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
