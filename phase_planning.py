"""
Research-Based Phase Planning Engine for Body Composition Forecasting

This module implements the sophisticated phase planning system that determines
optimal bulk/cut sequences based on scientific research and evidence-based
protocols from MacroFactor, Stronger By Science, and peer-reviewed studies.

Research Foundation:
- MacroFactor rate recommendations based on training level analysis
- Stronger By Science flexible template philosophy
- Forbes & Hall P-ratio research on lean vs fat gain/loss ratios
- Body fat threshold research for sustainable cutting and bulking

Key Citations:
- MacroFactor (2023): "Training Level-Specific Rate Recommendations"
- Helms et al. (2014): "Evidence-based recommendations for natural bodybuilding contest preparation"
- Forbes & Hall (2007): "Body composition changes during weight loss"
- Garthe et al. (2011): "Effect of two different weight-loss rates on body composition"
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from shared_models import (
    BF_THRESHOLDS,
    BFRangeConfig,
    PhaseConfig,
    PhaseSequence,
    PhaseTransition,
    PhaseType,
    TemplateType,
    TrainingLevel,
    UserProfile,
)

logger = logging.getLogger(__name__)


class PhaseConfigError(Exception):
    """Raised when phase configuration violates research-based safety constraints"""

    pass


class TransitionError(Exception):
    """Raised when phase transition logic fails validation"""

    pass


class RateCalculationError(Exception):
    """Raised when rate calculation produces values outside research bounds"""

    pass


class RateCalculator:
    """
    Evidence-based rate calculator implementing MacroFactor research.

    Research Source: MacroFactor (2023) training level analysis:
    - Novice: Higher rates due to rapid adaptation potential
    - Intermediate: Moderate rates balancing progress and sustainability
    - Advanced: Conservative rates due to adaptation limitations

    Body weight scaling based on percentage calculations ensures proper
    scaling across different body sizes (Hall et al., 2011).
    """

    # MacroFactor validated rates (% body weight per week)
    BULK_RATES = {
        TrainingLevel.NOVICE: {
            "conservative": 0.2,
            "moderate": 0.5,  # Happy medium - research validated
            "aggressive": 0.8,
        },
        TrainingLevel.INTERMEDIATE: {
            "conservative": 0.15,
            "moderate": 0.325,  # Happy medium - research validated
            "aggressive": 0.575,
        },
        TrainingLevel.ADVANCED: {
            "conservative": 0.1,
            "moderate": 0.15,  # Happy medium - research validated
            "aggressive": 0.35,
        },
    }

    # Universal cutting rates based on Garthe et al. (2011) research
    CUT_RATES = {
        "conservative": 0.25,  # Minimal muscle loss
        "moderate": 0.625,  # 0.5-0.75% average - optimal balance
        "aggressive": 1.0,  # Higher muscle loss risk
    }

    # Age adjustment factors based on metabolic research
    AGE_FACTORS = {
        "under_30": 1.0,
        "30_to_40": 0.9,
        "40_to_50": 0.8,
        "over_50": 0.7,
    }

    def get_bulk_rate(
        self, training_level: TrainingLevel, aggressiveness: str = "moderate"
    ) -> float:
        """
        Get evidence-based bulking rate for training level.

        Args:
            training_level: User's training experience
            aggressiveness: Rate preference (conservative/moderate/aggressive)

        Returns:
            Rate as percentage of body weight per week

        Raises:
            RateCalculationError: If parameters are invalid
        """
        if training_level not in self.BULK_RATES:
            raise RateCalculationError(f"Invalid training level: {training_level}")

        if aggressiveness not in self.BULK_RATES[training_level]:
            raise RateCalculationError(f"Invalid aggressiveness: {aggressiveness}")

        rate = self.BULK_RATES[training_level][aggressiveness]
        logger.info(
            f"Bulk rate for {training_level.value} {aggressiveness}: {rate}% BW/week"
        )
        return rate

    def get_cut_rate(self, aggressiveness: str = "moderate") -> float:
        """
        Get evidence-based cutting rate.

        Based on Garthe et al. (2011) research showing optimal muscle retention
        with moderate rates around 0.5-0.75% body weight per week.

        Args:
            aggressiveness: Rate preference

        Returns:
            Rate as percentage of body weight per week
        """
        if aggressiveness not in self.CUT_RATES:
            raise RateCalculationError(f"Invalid cut aggressiveness: {aggressiveness}")

        rate = self.CUT_RATES[aggressiveness]
        logger.info(f"Cut rate {aggressiveness}: {rate}% BW/week")
        return rate

    def apply_body_weight_scaling(self, base_rate: float, weight_lbs: float) -> float:
        """
        Apply body weight scaling to percentage-based rate.

        Based on Hall et al. (2011) energy balance calculations.

        Args:
            base_rate: Rate as percentage of body weight per week
            weight_lbs: Current body weight in pounds

        Returns:
            Absolute weight change in pounds per week
        """
        if base_rate <= 0 or weight_lbs <= 0:
            raise RateCalculationError("Rate and weight must be positive")

        absolute_rate = weight_lbs * (base_rate / 100)
        logger.debug(
            f"Weight scaling: {base_rate}% of {weight_lbs} lbs = {absolute_rate} lbs/week"
        )
        return absolute_rate

    def apply_age_adjustment(self, base_rate: float, age: float) -> float:
        """
        Apply age-based adjustment to rates.

        Based on metabolic research showing decreased adaptation rates with age.

        Args:
            base_rate: Unadjusted rate
            age: User's age

        Returns:
            Age-adjusted rate
        """
        if age < 30:
            factor = self.AGE_FACTORS["under_30"]
        elif age < 40:
            factor = self.AGE_FACTORS["30_to_40"]
        elif age < 50:
            factor = self.AGE_FACTORS["40_to_50"]
        else:
            factor = self.AGE_FACTORS["over_50"]

        adjusted_rate = base_rate * factor
        logger.info(
            f"Age adjustment: {base_rate} * {factor} = {adjusted_rate} (age {age})"
        )
        return adjusted_rate

    def get_p_ratio(
        self, phase_type: PhaseType, body_fat_pct: float, gender: str
    ) -> float:
        """
        Calculate P-ratio (proportion of weight change as lean mass).

        Based on Forbes & Hall research and updated with Stronger By Science analysis:
        - Bulking: ~45-50% lean mass regardless of body fat
        - Cutting: Higher P-ratio (more muscle retention) at higher body fat

        Args:
            phase_type: Current phase (cut/bulk)
            body_fat_pct: Current body fat percentage
            gender: User's gender for threshold determination

        Returns:
            P-ratio (0.0-1.0)
        """
        if phase_type == PhaseType.BULK:
            # Forbes research: bulking P-ratio relatively stable
            return 0.475  # Mid-point of 45-50% range

        elif phase_type == PhaseType.CUT:
            # Body fat dependent P-ratio for cutting (Hall et al.)
            thresholds = BF_THRESHOLDS[gender.lower()]

            if body_fat_pct > thresholds["healthy_max"]:
                # High body fat: better muscle retention (20-25%)
                return 0.225
            elif body_fat_pct > thresholds["acceptable_max"]:
                # Moderate body fat: intermediate retention (30-40%)
                return 0.35
            else:
                # Low body fat: higher muscle loss risk (30-40%)
                return 0.35

        else:  # MAINTENANCE
            return 0.5  # Balanced maintenance


class PhaseTemplateEngine:
    """
    Intelligent template selection based on research-backed decision trees.

    Implements MacroFactor decision logic and Stronger By Science philosophy:
    - Cut-first for users above healthy body fat thresholds
    - Bulk-first option for lean users prioritizing muscle gain
    - Safety-first approach based on health research
    """

    def __init__(self, rate_calculator: RateCalculator):
        self.rate_calculator = rate_calculator

    def _estimate_effective_max_bf_from_weight(
        self, user_profile: UserProfile, max_weight_lbs: float
    ) -> float:
        """
        Estimate the effective maximum body fat percentage given a weight constraint.
        
        This helps template generation set realistic BF% targets when weight constraints are active.
        
        Args:
            user_profile: User profile with current body composition
            max_weight_lbs: Maximum allowed weight
            
        Returns:
            Estimated maximum body fat percentage at the weight limit
        """
        try:
            # Get current body composition
            latest_scan = user_profile.scan_history[-1]
            if hasattr(latest_scan, "total_lean_mass_lbs"):
                current_lean_lbs = latest_scan.total_lean_mass_lbs
            else:
                current_lean_lbs = latest_scan["total_lean_mass_lbs"]

            # Conservative estimate: assume lean mass stays roughly constant during weight gain
            # This gives us a conservative upper bound on BF%
            # max_weight = lean_mass + fat_mass
            # max_fat_mass = max_weight - lean_mass
            # max_bf_pct = max_fat_mass / max_weight * 100

            max_fat_lbs = max_weight_lbs - current_lean_lbs
            if max_fat_lbs <= 0:
                # Weight constraint is below current lean mass - very restrictive
                return 5.0  # Very low BF% - essentially a cutting scenario

            estimated_max_bf_pct = (max_fat_lbs / max_weight_lbs) * 100

            # Cap at reasonable maximums
            gender = user_profile.gender.lower()
            absolute_max = 35 if gender == "female" else 25

            return min(estimated_max_bf_pct, absolute_max)

        except Exception:
            # If calculation fails, return a conservative estimate
            return 15.0

    def _calculate_effective_bulk_target(
        self, user_profile: UserProfile, desired_bf_pct: float, bf_range_config
    ) -> tuple[float, str]:
        """
        Calculate effective bulk target considering both BF% desires and weight constraints.
        
        Args:
            user_profile: User profile data
            desired_bf_pct: Desired BF% target from template logic
            bf_range_config: BF range configuration with potential weight constraint
            
        Returns:
            tuple: (effective_target_bf_pct, rationale_suffix)
        """
        if bf_range_config is None or bf_range_config.max_weight_lbs is None:
            # No weight constraint, use desired BF%
            return desired_bf_pct, ""

        # Calculate weight-constrained max BF%
        weight_constrained_max_bf = self._estimate_effective_max_bf_from_weight(
            user_profile, bf_range_config.max_weight_lbs
        )

        if weight_constrained_max_bf < desired_bf_pct:
            # Weight constraint is more restrictive than desired BF%
            return weight_constrained_max_bf, (
                f" (limited by weight constraint of {bf_range_config.max_weight_lbs} lbs)"
            )
        else:
            # Desired BF% is achievable within weight constraint
            return desired_bf_pct, ""

    def select_template(
        self, user_profile: UserProfile, bf_range_config=None
    ) -> TemplateType:
        """
        Select optimal template based on current body composition.

        Decision tree based on:
        - Health research: Cut recommended above 25%M/35%F (Helms et al., 2014)
        - Safety research: Prioritize sustainable body fat ranges
        - Stronger By Science: Allow flexibility for experienced users
        - Custom BF range: Suggest template based on position relative to custom range

        Args:
            user_profile: User's demographic and body composition data
            bf_range_config: Optional custom BF range configuration

        Returns:
            Recommended template type
        """
        from shared_models import BFRangeConfig

        latest_scan = user_profile.scan_history[-1]
        current_bf = latest_scan.body_fat_percentage
        gender = user_profile.gender.lower()
        thresholds = BF_THRESHOLDS[gender]

        # If custom BF range provided, use it for template suggestion
        if bf_range_config is not None:
            return self._select_template_with_custom_range(
                current_bf, gender, bf_range_config
            )

        # Research-based template selection (existing logic)
        if current_bf > thresholds["healthy_max"]:
            logger.info(
                f"Cut-first template selected: BF {current_bf:.1f}% > {thresholds['healthy_max']}% "
                f"(health threshold for {gender})"
            )
            return TemplateType.CUT_FIRST
        else:
            logger.info(
                f"Bulk-first template available: BF {current_bf:.1f}% <= {thresholds['healthy_max']}% "
                f"(healthy range for {gender})"
            )
            # Default to CUT_FIRST for safety, but BULK_FIRST is viable
            return TemplateType.CUT_FIRST

    def _select_template_with_custom_range(
        self, current_bf: float, gender: str, bf_range_config
    ) -> TemplateType:
        """
        Select template based on position relative to custom BF range.

        Args:
            current_bf: Current body fat percentage
            gender: User's gender for safety checks
            bf_range_config: Custom BF range configuration

        Returns:
            Recommended template type
        """
        from shared_models import BFRangeConfig

        # Safety check: ensure custom range respects minimums
        safety_min = BF_THRESHOLDS[gender]["minimum"]
        if bf_range_config.min_bf_pct < safety_min:
            logger.warning(
                f"Custom min BF {bf_range_config.min_bf_pct}% below safety minimum {safety_min}% for {gender}"
            )

        # Template suggestion based on position relative to custom range
        if current_bf > bf_range_config.max_bf_pct:
            logger.info(
                f"Cut-first template suggested: BF {current_bf:.1f}% > custom max {bf_range_config.max_bf_pct}%"
            )
            return TemplateType.CUT_FIRST
        elif current_bf < bf_range_config.min_bf_pct:
            # Check if bulking is safe
            if current_bf <= safety_min + 2:
                logger.info(
                    f"Bulk-first template suggested: BF {current_bf:.1f}% near safety minimum {safety_min}%"
                )
                return TemplateType.BULK_FIRST
            else:
                logger.info(
                    f"Bulk-first template suggested: BF {current_bf:.1f}% < custom min {bf_range_config.min_bf_pct}%"
                )
                return TemplateType.BULK_FIRST
        else:
            logger.info(
                f"Within custom range {bf_range_config.min_bf_pct}-{bf_range_config.max_bf_pct}%: "
                f"Cut-first template as default (user can override)"
            )
            return TemplateType.CUT_FIRST

    def generate_sequence(
        self,
        template: TemplateType,
        user_profile: UserProfile,
        target_bf_pct: Optional[float] = None,
        bf_range_config=None,
    ) -> PhaseSequence:
        """
        Generate complete phase sequence for template.

        Args:
            template: Selected template type
            user_profile: User's profile data
            target_bf_pct: Optional target body fat percentage
            bf_range_config: Optional custom BF range configuration

        Returns:
            Complete phase sequence with research rationale
        """
        if template == TemplateType.CUT_FIRST:
            return self._generate_cut_first_sequence(
                user_profile, target_bf_pct, bf_range_config
            )
        elif template == TemplateType.BULK_FIRST:
            return self._generate_bulk_first_sequence(
                user_profile, target_bf_pct, bf_range_config
            )
        else:
            raise PhaseConfigError(f"Unknown template: {template}")

    def _generate_cut_first_sequence(
        self,
        user_profile: UserProfile,
        target_bf_pct: Optional[float],
        bf_range_config=None,
    ) -> PhaseSequence:
        """
        Generate cut-first template sequence with optional custom BF range.

        Logic:
        1. If custom range: Use custom boundaries for cycling
        2. If outside custom range: Get into range first, then cycle within it
        3. If no custom range: Use research-backed thresholds (existing behavior)
        """
        gender = user_profile.gender.lower()
        thresholds = BF_THRESHOLDS[gender]
        latest_scan = user_profile.scan_history[-1]
        current_bf = latest_scan.body_fat_percentage

        # Calculate rates once for reuse
        cut_rate = self.rate_calculator.get_cut_rate("moderate")
        bulk_rate = self.rate_calculator.get_bulk_rate(
            user_profile.training_level, "moderate"
        )

        # Use custom range if provided
        if bf_range_config is not None:
            return self._generate_cut_first_with_custom_range(
                user_profile, current_bf, bf_range_config, cut_rate, bulk_rate
            )

        # Original research-based logic when no custom range
        phases = []

        # Phase 1: Initial cut to healthy range
        if current_bf > thresholds["acceptable_max"]:
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.CUT,
                    target_bf_pct=thresholds["acceptable_max"],
                    min_duration_weeks=8,  # Garthe et al. minimum
                    max_duration_weeks=24,  # Practical limit
                    rate_pct_per_week=cut_rate,
                    rationale=f"Initial cut to {thresholds['acceptable_max']}% based on health research",
                )
            )

        # Phase 2: Bulk to upper acceptable limit (considering weight constraints)
        bulk_target_bf = thresholds["acceptable_max"] + 3  # Default conservative buffer
        bulk_rationale = f"Bulk to {bulk_target_bf}% for muscle gain"

        # Adjust bulk target if weight constraint is active
        if bf_range_config is not None and bf_range_config.max_weight_lbs is not None:
            weight_constrained_max_bf = self._estimate_effective_max_bf_from_weight(
                user_profile, bf_range_config.max_weight_lbs
            )
            if weight_constrained_max_bf < bulk_target_bf:
                bulk_target_bf = weight_constrained_max_bf
                bulk_rationale = (
                    f"Bulk to {bulk_target_bf:.1f}% (limited by weight constraint "
                    f"of {bf_range_config.max_weight_lbs} lbs)"
                )

        phases.append(
            PhaseConfig(
                phase_type=PhaseType.BULK,
                target_bf_pct=bulk_target_bf,
                min_duration_weeks=12,  # Muscle protein synthesis adaptation
                max_duration_weeks=36,  # Sustainability limit
                rate_pct_per_week=bulk_rate,
                rationale=bulk_rationale,
            )
        )

        # Phase 3: Cut to preferred range
        phases.append(
            PhaseConfig(
                phase_type=PhaseType.CUT,
                target_bf_pct=thresholds["preferred_max"],
                min_duration_weeks=8,
                max_duration_weeks=20,
                rate_pct_per_week=cut_rate,
                rationale=f"Cut to preferred {thresholds['preferred_max']}% range",
            )
        )

        total_duration = sum(p.min_duration_weeks for p in phases)

        return PhaseSequence(
            template=TemplateType.CUT_FIRST,
            phases=phases,
            rationale=(
                "Cut-first template prioritizes reaching healthy body fat range first "
                "(Helms et al., 2014), then implements sustainable bulk/cut cycles."
            ),
            expected_duration_weeks=total_duration,
            safety_validated=True,
        )

    def _generate_cut_first_with_custom_range(
        self,
        user_profile: UserProfile,
        current_bf: float,
        bf_range_config,
        cut_rate: float,
        bulk_rate: float,
    ) -> PhaseSequence:
        """Generate cut-first sequence using custom BF range boundaries"""
        gender = user_profile.gender.lower()
        phases = []

        # Safety validation
        safety_min = BF_THRESHOLDS[gender]["minimum"]
        effective_min = max(bf_range_config.min_bf_pct, safety_min)
        if effective_min != bf_range_config.min_bf_pct:
            logger.warning(
                f"Adjusted custom min BF from {bf_range_config.min_bf_pct}% to {effective_min}% for safety"
            )

        # Build phases based on starting position and cut-first approach
        if current_bf > bf_range_config.max_bf_pct:
            # Above range: Cut to minimum, then cycle
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.CUT,
                    target_bf_pct=effective_min,
                    min_duration_weeks=8,
                    max_duration_weeks=24,
                    rate_pct_per_week=cut_rate,
                    rationale=f"Cut to custom range minimum {effective_min}% (cut-first approach)",
                )
            )
            # Calculate effective bulk target considering weight constraints
            bulk_target_bf, weight_suffix = self._calculate_effective_bulk_target(
                user_profile, bf_range_config.max_bf_pct, bf_range_config
            )
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.BULK,
                    target_bf_pct=bulk_target_bf,
                    min_duration_weeks=12,
                    max_duration_weeks=36,
                    rate_pct_per_week=bulk_rate,
                    rationale=f"Bulk to custom range maximum {bulk_target_bf:.1f}%{weight_suffix}",
                )
            )
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.CUT,
                    target_bf_pct=effective_min,
                    min_duration_weeks=8,
                    max_duration_weeks=20,
                    rate_pct_per_week=cut_rate,
                    rationale=f"Cut back to custom range minimum {effective_min}%",
                )
            )
        elif current_bf < effective_min:
            # Below range: Bulk directly to maximum (honoring cut-first by ending at minimum)
            # Calculate effective bulk target considering weight constraints
            bulk_target_bf, weight_suffix = self._calculate_effective_bulk_target(
                user_profile, bf_range_config.max_bf_pct, bf_range_config
            )
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.BULK,
                    target_bf_pct=bulk_target_bf,
                    min_duration_weeks=12,
                    max_duration_weeks=36,
                    rate_pct_per_week=bulk_rate,
                    rationale=f"Bulk to custom range maximum {bulk_target_bf:.1f}% (from below range){weight_suffix}",
                )
            )
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.CUT,
                    target_bf_pct=effective_min,
                    min_duration_weeks=8,
                    max_duration_weeks=20,
                    rate_pct_per_week=cut_rate,
                    rationale=f"Cut to custom range minimum {effective_min}% (cut-first goal)",
                )
            )
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.BULK,
                    target_bf_pct=bf_range_config.max_bf_pct,
                    min_duration_weeks=12,
                    max_duration_weeks=36,
                    rate_pct_per_week=bulk_rate,
                    rationale=f"Bulk back to custom range maximum {bf_range_config.max_bf_pct}%",
                )
            )
        else:
            # Within range: Cut to minimum first, then cycle
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.CUT,
                    target_bf_pct=effective_min,
                    min_duration_weeks=8,
                    max_duration_weeks=20,
                    rate_pct_per_week=cut_rate,
                    rationale=f"Cut to custom range minimum {effective_min}% (within range, cut-first)",
                )
            )
            # Calculate effective bulk target considering weight constraints
            bulk_target_bf, weight_suffix = self._calculate_effective_bulk_target(
                user_profile, bf_range_config.max_bf_pct, bf_range_config
            )
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.BULK,
                    target_bf_pct=bulk_target_bf,
                    min_duration_weeks=12,
                    max_duration_weeks=36,
                    rate_pct_per_week=bulk_rate,
                    rationale=f"Bulk to custom range maximum {bulk_target_bf:.1f}%{weight_suffix}",
                )
            )
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.CUT,
                    target_bf_pct=effective_min,
                    min_duration_weeks=8,
                    max_duration_weeks=20,
                    rate_pct_per_week=cut_rate,
                    rationale=f"Cut back to custom range minimum {effective_min}%",
                )
            )

        total_duration = sum(p.min_duration_weeks for p in phases)

        return PhaseSequence(
            template=TemplateType.CUT_FIRST,
            phases=phases,
            rationale=(
                f"Cut-first template with custom range {effective_min}-{bf_range_config.max_bf_pct}%. "
                f"Prioritizes cutting approach within user-defined aesthetic boundaries."
            ),
            expected_duration_weeks=total_duration,
            safety_validated=True,
        )

    def _generate_bulk_first_sequence(
        self,
        user_profile: UserProfile,
        target_bf_pct: Optional[float],
        bf_range_config=None,
    ) -> PhaseSequence:
        """
        Generate bulk-first template sequence with optional custom BF range.

        Logic:
        1. If custom range: Use custom boundaries for cycling
        2. If outside custom range: Get into range first, then cycle within it
        3. If no custom range: Use research-backed thresholds (existing behavior)
        """
        gender = user_profile.gender.lower()
        thresholds = BF_THRESHOLDS[gender]
        latest_scan = user_profile.scan_history[-1]
        current_bf = latest_scan.body_fat_percentage

        # Calculate rates once for reuse
        cut_rate = self.rate_calculator.get_cut_rate("moderate")
        bulk_rate = self.rate_calculator.get_bulk_rate(
            user_profile.training_level, "moderate"
        )

        # Use custom range if provided
        if bf_range_config is not None:
            return self._generate_bulk_first_with_custom_range(
                user_profile, current_bf, bf_range_config, cut_rate, bulk_rate
            )

        # Original research-based logic when no custom range
        phases = []

        # Phase 1: Initial bulk (considering weight constraints)
        target_bulk_bf = min(current_bf + 5, thresholds["acceptable_max"] + 2)
        bulk_rationale = f"Initial bulk to {target_bulk_bf}% for muscle prioritization"

        # Adjust bulk target if weight constraint is active
        if bf_range_config is not None and bf_range_config.max_weight_lbs is not None:
            weight_constrained_max_bf = self._estimate_effective_max_bf_from_weight(
                user_profile, bf_range_config.max_weight_lbs
            )
            if weight_constrained_max_bf < target_bulk_bf:
                target_bulk_bf = weight_constrained_max_bf
                bulk_rationale = (
                    f"Initial bulk to {target_bulk_bf:.1f}% (limited by weight constraint "
                    f"of {bf_range_config.max_weight_lbs} lbs)"
                )

        phases.append(
            PhaseConfig(
                phase_type=PhaseType.BULK,
                target_bf_pct=target_bulk_bf,
                min_duration_weeks=12,
                max_duration_weeks=36,
                rate_pct_per_week=bulk_rate,
                rationale=bulk_rationale,
            )
        )

        # Phase 2: Cut to preferred range
        phases.append(
            PhaseConfig(
                phase_type=PhaseType.CUT,
                target_bf_pct=thresholds["preferred_max"],
                min_duration_weeks=8,
                max_duration_weeks=20,
                rate_pct_per_week=cut_rate,
                rationale=f"Cut to preferred {thresholds['preferred_max']}% range",
            )
        )

        total_duration = sum(p.min_duration_weeks for p in phases)

        return PhaseSequence(
            template=TemplateType.BULK_FIRST,
            phases=phases,
            rationale=(
                "Bulk-first template for lean users prioritizing muscle gain "
                "(Stronger By Science flexible approach). Suitable for users "
                "already in healthy body fat range."
            ),
            expected_duration_weeks=total_duration,
            safety_validated=True,
        )

    def _generate_bulk_first_with_custom_range(
        self,
        user_profile: UserProfile,
        current_bf: float,
        bf_range_config,
        cut_rate: float,
        bulk_rate: float,
    ) -> PhaseSequence:
        """Generate bulk-first sequence using custom BF range boundaries"""
        gender = user_profile.gender.lower()
        phases = []

        # Safety validation
        safety_min = BF_THRESHOLDS[gender]["minimum"]
        effective_min = max(bf_range_config.min_bf_pct, safety_min)
        if effective_min != bf_range_config.min_bf_pct:
            logger.warning(
                f"Adjusted custom min BF from {bf_range_config.min_bf_pct}% to {effective_min}% for safety"
            )

        # Phase 1: Get into custom range if needed (Bulk-first approach)
        if current_bf < effective_min:
            # Bulk to custom range maximum (bulk-first approach)
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.BULK,
                    target_bf_pct=bf_range_config.max_bf_pct,
                    min_duration_weeks=12,
                    max_duration_weeks=36,
                    rate_pct_per_week=bulk_rate,
                    rationale=f"Bulk to custom range maximum {bf_range_config.max_bf_pct}% (bulk-first approach)",
                )
            )
            # Follow with cut to minimum
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.CUT,
                    target_bf_pct=effective_min,
                    min_duration_weeks=8,
                    max_duration_weeks=20,
                    rate_pct_per_week=cut_rate,
                    rationale=f"Cut to custom range minimum {effective_min}%",
                )
            )
        elif current_bf > bf_range_config.max_bf_pct:
            # User chose bulk-first but is above range - still honor their choice by going higher first
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.BULK,
                    target_bf_pct=min(
                        current_bf + 5, bf_range_config.max_bf_pct + 5
                    ),  # Go higher first
                    min_duration_weeks=12,
                    max_duration_weeks=36,
                    rate_pct_per_week=bulk_rate,
                    rationale="Initial bulk (user chose bulk-first despite high BF)",
                )
            )
            # Then cut to custom range minimum
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.CUT,
                    target_bf_pct=effective_min,
                    min_duration_weeks=8,
                    max_duration_weeks=24,
                    rate_pct_per_week=cut_rate,
                    rationale=f"Cut to custom range minimum {effective_min}%",
                )
            )
        else:
            # Within range: bulk to maximum first (bulk-first approach)
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.BULK,
                    target_bf_pct=bf_range_config.max_bf_pct,
                    min_duration_weeks=12,
                    max_duration_weeks=36,
                    rate_pct_per_week=bulk_rate,
                    rationale=f"Bulk to custom range maximum {bf_range_config.max_bf_pct}% (within range, bulk-first)",
                )
            )
            # Follow with cut to minimum
            phases.append(
                PhaseConfig(
                    phase_type=PhaseType.CUT,
                    target_bf_pct=effective_min,
                    min_duration_weeks=8,
                    max_duration_weeks=20,
                    rate_pct_per_week=cut_rate,
                    rationale=f"Cut to custom range minimum {effective_min}%",
                )
            )

        # Final phase: Bulk back to custom range maximum (complete the cycle)
        phases.append(
            PhaseConfig(
                phase_type=PhaseType.BULK,
                target_bf_pct=bf_range_config.max_bf_pct,
                min_duration_weeks=12,
                max_duration_weeks=36,
                rate_pct_per_week=bulk_rate,
                rationale=f"Bulk back to custom range maximum {bf_range_config.max_bf_pct}%",
            )
        )

        total_duration = sum(p.min_duration_weeks for p in phases)

        return PhaseSequence(
            template=TemplateType.BULK_FIRST,
            phases=phases,
            rationale=(
                f"Bulk-first template with custom range {effective_min}-{bf_range_config.max_bf_pct}%. "
                f"Prioritizes bulking approach within user-defined aesthetic boundaries."
            ),
            expected_duration_weeks=total_duration,
            safety_validated=True,
        )


class PhaseTransitionManager:
    """
    Advanced phase transition management with research-backed logic.

    Implements sophisticated transition rules based on:
    - Body fat thresholds from health research
    - Minimum duration requirements from adaptation studies
    - Safety validation from sustainable dieting research
    """

    def __init__(self, rate_calculator: RateCalculator):
        self.rate_calculator = rate_calculator

    def should_transition(
        self,
        current_bf_pct: float,
        current_phase: PhaseType,
        phase_config: PhaseConfig,
        weeks_in_phase: int,
        gender: str,
        current_weight_lbs: Optional[float] = None,
        bf_range_config: Optional["BFRangeConfig"] = None,
    ) -> bool:
        """
        Determine if phase transition should occur.

        Args:
            current_bf_pct: Current body fat percentage
            current_phase: Current phase type
            phase_config: Current phase configuration
            weeks_in_phase: Duration in current phase
            gender: User's gender for threshold lookup
            current_weight_lbs: Current weight in pounds (for weight constraint checks)
            bf_range_config: BF range config with potential max weight constraint

        Returns:
            True if transition should occur
        """
        # Check minimum duration requirement (adaptation research)
        if weeks_in_phase < phase_config.min_duration_weeks:
            logger.debug(
                f"No transition: {weeks_in_phase} weeks < {phase_config.min_duration_weeks} minimum"
            )
            return False

        # Check target achievement
        if current_phase == PhaseType.CUT:
            target_reached = current_bf_pct <= phase_config.target_bf_pct
        elif current_phase == PhaseType.BULK:
            target_reached = current_bf_pct >= phase_config.target_bf_pct
        else:  # MAINTENANCE
            target_reached = False

        if target_reached:
            logger.info(
                f"Phase transition triggered: BF {current_bf_pct:.1f}% reached "
                f"{phase_config.target_bf_pct:.1f}% target"
            )
            return True

        # Check weight constraint (force cut if weight limit exceeded during bulk)
        if (current_weight_lbs is not None and
            bf_range_config is not None and
            bf_range_config.max_weight_lbs is not None):

            if (current_phase == PhaseType.BULK and
                current_weight_lbs >= bf_range_config.max_weight_lbs):
                logger.info(
                    f"Phase transition triggered: Weight {current_weight_lbs:.1f} lbs "
                    f"reached maximum {bf_range_config.max_weight_lbs:.1f} lbs (forcing cut)"
                )
                return True

        # Check maximum duration (sustainability research)
        if weeks_in_phase >= phase_config.max_duration_weeks:
            logger.warning(
                f"Phase transition forced: {weeks_in_phase} weeks >= "
                f"{phase_config.max_duration_weeks} maximum"
            )
            return True

        return False

    def get_next_phase_config(
        self,
        current_phase: PhaseConfig,
        user_profile: UserProfile,
        sequence: PhaseSequence,
    ) -> Optional[PhaseConfig]:
        """
        Get next phase configuration from sequence.

        Args:
            current_phase: Current phase configuration
            user_profile: User profile data
            sequence: Complete phase sequence

        Returns:
            Next phase configuration or None if sequence complete
        """
        try:
            current_index = sequence.phases.index(current_phase)
            if current_index + 1 < len(sequence.phases):
                next_phase = sequence.phases[current_index + 1]
                logger.info(
                    f"Next phase: {next_phase.phase_type.value} to {next_phase.target_bf_pct}%"
                )
                return next_phase
        except (ValueError, IndexError):
            logger.warning("Could not find current phase in sequence")

        return None

    def validate_transition(
        self,
        from_phase: PhaseType,
        to_phase: PhaseType,
        current_bf_pct: float,
        gender: str,
    ) -> bool:
        """
        Validate phase transition against safety research.

        Args:
            from_phase: Source phase
            to_phase: Target phase
            current_bf_pct: Current body fat percentage
            gender: User's gender

        Returns:
            True if transition is safe

        Raises:
            TransitionError: If transition violates safety constraints
        """
        thresholds = BF_THRESHOLDS[gender.lower()]

        # Safety bounds validation (health research)
        if current_bf_pct < thresholds["minimum"]:
            raise TransitionError(
                f"Unsafe transition: BF {current_bf_pct:.1f}% below minimum "
                f"{thresholds['minimum']}% for {gender}"
            )

        if to_phase == PhaseType.CUT and current_bf_pct <= thresholds["minimum"] + 2:
            raise TransitionError(
                f"Unsafe cut initiation: BF {current_bf_pct:.1f}% too close to "
                f"minimum {thresholds['minimum']}% for {gender}"
            )

        # Logical transition validation
        valid_transitions = {
            PhaseType.CUT: [PhaseType.BULK, PhaseType.MAINTENANCE],
            PhaseType.BULK: [PhaseType.CUT, PhaseType.MAINTENANCE],
            PhaseType.MAINTENANCE: [PhaseType.CUT, PhaseType.BULK],
        }

        if to_phase not in valid_transitions[from_phase]:
            raise TransitionError(
                f"Invalid transition: {from_phase.value} -> {to_phase.value}"
            )

        logger.info(f"Transition validated: {from_phase.value} -> {to_phase.value}")
        return True


class PhaseValidationEngine:
    """
    Comprehensive validation engine for phase configurations.

    Ensures all phase plans conform to research-based safety constraints
    and practical limitations.
    """

    @staticmethod
    def validate_phase_config(config: PhaseConfig, gender: str) -> bool:
        """
        Validate individual phase configuration.

        Args:
            config: Phase configuration to validate
            gender: User's gender for threshold lookup

        Returns:
            True if configuration is valid

        Raises:
            PhaseConfigError: If configuration is invalid
        """
        thresholds = BF_THRESHOLDS[gender.lower()]

        # Duration validation
        if config.min_duration_weeks < 6:
            raise PhaseConfigError("Minimum duration must be at least 6 weeks")
        if config.max_duration_weeks > 52:
            raise PhaseConfigError("Maximum duration should not exceed 52 weeks")
        if config.min_duration_weeks >= config.max_duration_weeks:
            raise PhaseConfigError("Minimum duration must be less than maximum")

        # Body fat target validation
        if config.target_bf_pct < thresholds["minimum"]:
            raise PhaseConfigError(
                f"Target BF {config.target_bf_pct}% below safe minimum "
                f"{thresholds['minimum']}% for {gender}"
            )

        # Rate validation
        if config.rate_pct_per_week <= 0:
            raise PhaseConfigError("Rate must be positive")
        if config.rate_pct_per_week > 1.5:
            raise PhaseConfigError("Rate exceeds sustainable limit of 1.5% BW/week")

        return True

    @staticmethod
    def validate_sequence(sequence: PhaseSequence, gender: str) -> bool:
        """
        Validate complete phase sequence.

        Args:
            sequence: Phase sequence to validate
            gender: User's gender

        Returns:
            True if sequence is valid

        Raises:
            PhaseConfigError: If sequence is invalid
        """
        if not sequence.phases:
            raise PhaseConfigError("Sequence must contain at least one phase")

        # Validate each phase
        for phase in sequence.phases:
            PhaseValidationEngine.validate_phase_config(phase, gender)

        # Validate sequence logic
        for i in range(len(sequence.phases) - 1):
            current = sequence.phases[i]
            next_phase = sequence.phases[i + 1]

            # Ensure alternating phases
            if current.phase_type == next_phase.phase_type:
                raise PhaseConfigError(
                    f"Consecutive phases cannot be the same type: "
                    f"{current.phase_type.value} -> {next_phase.phase_type.value}"
                )

        # Total duration validation
        total_duration = sum(p.max_duration_weeks for p in sequence.phases)
        if total_duration > 260:  # 5 years
            raise PhaseConfigError(
                f"Total sequence duration {total_duration} weeks exceeds "
                "practical limit of 5 years"
            )

        return True
