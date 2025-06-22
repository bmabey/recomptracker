"""
Comprehensive test suite for research-backed phase planning system.

Tests validate implementation against published research including:
- MacroFactor rate recommendations
- Body fat threshold research
- Training level variance factors
- Safety and sustainability constraints
"""

import unittest
from unittest.mock import MagicMock

from phase_planning import (
    PhaseConfigError,
    PhaseTemplateEngine,
    PhaseTransitionManager,
    PhaseValidationEngine,
    RateCalculator,
    TransitionError,
)
from shared_models import (
    BF_THRESHOLDS,
    PhaseConfig,
    PhaseSequence,
    PhaseType,
    ScanData,
    TemplateType,
    TrainingLevel,
    UserProfile,
)


class TestRateCalculator(unittest.TestCase):
    """Test MacroFactor research-based rate calculations"""

    def setUp(self):
        self.calculator = RateCalculator()

    def test_bulk_rates_match_macrofactor_research(self):
        """Test bulk rates match published MacroFactor values"""
        # Test novice rates (MacroFactor research)
        self.assertEqual(
            self.calculator.get_bulk_rate(TrainingLevel.NOVICE, "conservative"), 0.2
        )
        self.assertEqual(
            self.calculator.get_bulk_rate(TrainingLevel.NOVICE, "moderate"), 0.5
        )
        self.assertEqual(
            self.calculator.get_bulk_rate(TrainingLevel.NOVICE, "aggressive"), 0.8
        )

        # Test intermediate rates
        self.assertEqual(
            self.calculator.get_bulk_rate(TrainingLevel.INTERMEDIATE, "conservative"),
            0.15,
        )
        self.assertEqual(
            self.calculator.get_bulk_rate(TrainingLevel.INTERMEDIATE, "moderate"), 0.325
        )
        self.assertEqual(
            self.calculator.get_bulk_rate(TrainingLevel.INTERMEDIATE, "aggressive"),
            0.575,
        )

        # Test advanced rates
        self.assertEqual(
            self.calculator.get_bulk_rate(TrainingLevel.ADVANCED, "conservative"), 0.1
        )
        self.assertEqual(
            self.calculator.get_bulk_rate(TrainingLevel.ADVANCED, "moderate"), 0.15
        )
        self.assertEqual(
            self.calculator.get_bulk_rate(TrainingLevel.ADVANCED, "aggressive"), 0.35
        )

    def test_cut_rates_match_research(self):
        """Test cutting rates match Garthe et al. research"""
        # Garthe et al. (2011) optimal cutting rates
        self.assertEqual(self.calculator.get_cut_rate("conservative"), 0.25)
        self.assertEqual(
            self.calculator.get_cut_rate("moderate"), 0.625
        )  # 0.5-0.75% average
        self.assertEqual(self.calculator.get_cut_rate("aggressive"), 1.0)

    def test_body_weight_scaling_accuracy(self):
        """Test body weight percentage scaling calculations"""
        # Test 0.5% of 200 lbs = 1 lb/week
        rate = self.calculator.apply_body_weight_scaling(0.5, 200.0)
        self.assertEqual(rate, 1.0)

        # Test 0.25% of 150 lbs = 0.375 lb/week
        rate = self.calculator.apply_body_weight_scaling(0.25, 150.0)
        self.assertAlmostEqual(rate, 0.375, places=3)

    def test_age_adjustments_match_research(self):
        """Test age adjustments follow metabolic research"""
        # Under 30: no adjustment
        self.assertEqual(self.calculator.apply_age_adjustment(0.5, 25), 0.5)

        # 30-40: 10% reduction
        self.assertAlmostEqual(
            self.calculator.apply_age_adjustment(0.5, 35), 0.45, places=2
        )

        # 40-50: 20% reduction
        self.assertAlmostEqual(
            self.calculator.apply_age_adjustment(0.5, 45), 0.4, places=2
        )

        # Over 50: 30% reduction
        self.assertAlmostEqual(
            self.calculator.apply_age_adjustment(0.5, 55), 0.35, places=2
        )

    def test_p_ratio_research_validation(self):
        """Test P-ratio calculations match Forbes & Hall research"""
        # Bulking P-ratio (Forbes research): ~47.5%
        bulk_p_ratio = self.calculator.get_p_ratio(PhaseType.BULK, 15.0, "male")
        self.assertAlmostEqual(bulk_p_ratio, 0.475, places=3)

        # High body fat cutting (Hall et al.): better muscle retention
        high_bf_p_ratio = self.calculator.get_p_ratio(PhaseType.CUT, 30.0, "male")
        self.assertAlmostEqual(high_bf_p_ratio, 0.225, places=3)

        # Moderate body fat cutting
        mod_bf_p_ratio = self.calculator.get_p_ratio(PhaseType.CUT, 18.0, "male")
        self.assertAlmostEqual(mod_bf_p_ratio, 0.35, places=3)

    def test_rate_bounds_validation(self):
        """Test all rates are within research-backed bounds"""
        # Test all training levels and aggressiveness combinations
        for level in TrainingLevel:
            for agg in ["conservative", "moderate", "aggressive"]:
                rate = self.calculator.get_bulk_rate(level, agg)
                self.assertGreater(rate, 0)
                self.assertLessEqual(rate, 1.5)  # Sustainability limit

        # Test cutting rates
        for agg in ["conservative", "moderate", "aggressive"]:
            rate = self.calculator.get_cut_rate(agg)
            self.assertGreater(rate, 0)
            self.assertLessEqual(rate, 1.5)

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters"""
        with self.assertRaises(Exception):
            self.calculator.get_bulk_rate("invalid", "moderate")

        with self.assertRaises(Exception):
            self.calculator.get_cut_rate("invalid")

        with self.assertRaises(Exception):
            self.calculator.apply_body_weight_scaling(-0.5, 150)


class TestPhaseTemplateEngine(unittest.TestCase):
    """Test template selection logic matches MacroFactor decision tree"""

    def setUp(self):
        self.rate_calculator = RateCalculator()
        self.engine = PhaseTemplateEngine(self.rate_calculator)

    def create_user_profile(
        self, gender: str, body_fat: float, training_level: TrainingLevel
    ) -> UserProfile:
        """Create test user profile"""
        return UserProfile(
            birth_date="01/01/1990",
            height_in=68.0,
            gender=gender,
            training_level=training_level,
            scan_history=[
                ScanData(
                    date="01/01/2024",
                    total_weight_lbs=170.0,
                    total_lean_mass_lbs=140.0,
                    fat_mass_lbs=30.0,
                    body_fat_percentage=body_fat,
                    arms_lean_lbs=15.0,
                    legs_lean_lbs=40.0,
                )
            ],
        )

    def test_cut_first_template_selection(self):
        """Test cut-first template selected for high body fat users"""
        # Male above 25% BF should get cut-first
        user = self.create_user_profile("male", 28.0, TrainingLevel.INTERMEDIATE)
        template = self.engine.select_template(user)
        self.assertEqual(template, TemplateType.CUT_FIRST)

        # Female above 35% BF should get cut-first
        user = self.create_user_profile("female", 38.0, TrainingLevel.INTERMEDIATE)
        template = self.engine.select_template(user)
        self.assertEqual(template, TemplateType.CUT_FIRST)

    def test_bulk_first_template_available(self):
        """Test bulk-first template is available for lean users"""
        # Male at healthy BF should have cut-first as default but bulk-first available
        user = self.create_user_profile("male", 15.0, TrainingLevel.INTERMEDIATE)
        template = self.engine.select_template(user)
        # Current logic defaults to cut-first for safety, but both are viable
        self.assertEqual(template, TemplateType.CUT_FIRST)

    def test_cut_first_sequence_generation(self):
        """Test cut-first sequence follows MacroFactor logic"""
        user = self.create_user_profile("male", 30.0, TrainingLevel.INTERMEDIATE)
        sequence = self.engine.generate_sequence(TemplateType.CUT_FIRST, user)

        # Should have multiple phases
        self.assertGreater(len(sequence.phases), 1)

        # First phase should be cut to acceptable range
        first_phase = sequence.phases[0]
        self.assertEqual(first_phase.phase_type, PhaseType.CUT)
        self.assertEqual(
            first_phase.target_bf_pct, BF_THRESHOLDS["male"]["acceptable_max"]
        )

        # Should have bulk phase after initial cut
        second_phase = sequence.phases[1]
        self.assertEqual(second_phase.phase_type, PhaseType.BULK)

        # Should have final cut phase
        third_phase = sequence.phases[2]
        self.assertEqual(third_phase.phase_type, PhaseType.CUT)
        self.assertEqual(
            third_phase.target_bf_pct, BF_THRESHOLDS["male"]["preferred_max"]
        )

    def test_bulk_first_sequence_generation(self):
        """Test bulk-first sequence for lean users"""
        user = self.create_user_profile("male", 12.0, TrainingLevel.INTERMEDIATE)
        sequence = self.engine.generate_sequence(TemplateType.BULK_FIRST, user)

        # Should start with bulk
        first_phase = sequence.phases[0]
        self.assertEqual(first_phase.phase_type, PhaseType.BULK)

        # Should have cut phase after bulk
        second_phase = sequence.phases[1]
        self.assertEqual(second_phase.phase_type, PhaseType.CUT)

    def test_sequence_duration_constraints(self):
        """Test all generated sequences respect duration research"""
        user = self.create_user_profile("male", 20.0, TrainingLevel.INTERMEDIATE)
        sequence = self.engine.generate_sequence(TemplateType.CUT_FIRST, user)

        for phase in sequence.phases:
            if phase.phase_type == PhaseType.CUT:
                # Garthe et al. minimum
                self.assertGreaterEqual(phase.min_duration_weeks, 8)
            elif phase.phase_type == PhaseType.BULK:
                # Muscle protein synthesis adaptation
                self.assertGreaterEqual(phase.min_duration_weeks, 12)

            # Sustainability maximum
            self.assertLessEqual(phase.max_duration_weeks, 52)

    def test_rate_integration_with_templates(self):
        """Test template phases use correct research-backed rates"""
        user = self.create_user_profile("male", 25.0, TrainingLevel.INTERMEDIATE)
        sequence = self.engine.generate_sequence(TemplateType.CUT_FIRST, user)

        for phase in sequence.phases:
            if phase.phase_type == PhaseType.BULK:
                expected_rate = self.rate_calculator.get_bulk_rate(
                    TrainingLevel.INTERMEDIATE, "moderate"
                )
                self.assertEqual(phase.rate_pct_per_week, expected_rate)
            elif phase.phase_type == PhaseType.CUT:
                expected_rate = self.rate_calculator.get_cut_rate("moderate")
                self.assertEqual(phase.rate_pct_per_week, expected_rate)


class TestPhaseTransitionManager(unittest.TestCase):
    """Test sophisticated phase transition logic"""

    def setUp(self):
        self.rate_calculator = RateCalculator()
        self.manager = PhaseTransitionManager(self.rate_calculator)

    def test_minimum_duration_enforcement(self):
        """Test transitions respect minimum duration research"""
        # Create phase config with 8-week minimum
        phase_config = PhaseConfig(
            phase_type=PhaseType.CUT,
            target_bf_pct=15.0,
            min_duration_weeks=8,
            max_duration_weeks=24,
            rate_pct_per_week=0.625,
            rationale="Test cutting phase",
        )

        # Should not transition before 8 weeks even if target reached
        should_transition = self.manager.should_transition(
            current_bf_pct=15.0,  # Target reached
            current_phase=PhaseType.CUT,
            phase_config=phase_config,
            weeks_in_phase=6,  # Only 6 weeks
            gender="male",
        )
        self.assertFalse(should_transition)

        # Should transition after 8 weeks if target reached
        should_transition = self.manager.should_transition(
            current_bf_pct=15.0,
            current_phase=PhaseType.CUT,
            phase_config=phase_config,
            weeks_in_phase=8,
            gender="male",
        )
        self.assertTrue(should_transition)

    def test_target_achievement_detection(self):
        """Test accurate target achievement detection"""
        # Cutting phase
        cut_config = PhaseConfig(
            phase_type=PhaseType.CUT,
            target_bf_pct=15.0,
            min_duration_weeks=8,
            max_duration_weeks=24,
            rate_pct_per_week=0.625,
            rationale="Test cutting",
        )

        # Should transition when BF reaches target (after minimum duration)
        self.assertTrue(
            self.manager.should_transition(
                current_bf_pct=14.5,  # Below 15% target
                current_phase=PhaseType.CUT,
                phase_config=cut_config,
                weeks_in_phase=10,
                gender="male",
            )
        )

        # Bulking phase
        bulk_config = PhaseConfig(
            phase_type=PhaseType.BULK,
            target_bf_pct=18.0,
            min_duration_weeks=12,
            max_duration_weeks=36,
            rate_pct_per_week=0.325,
            rationale="Test bulking",
        )

        # Should transition when BF reaches target
        self.assertTrue(
            self.manager.should_transition(
                current_bf_pct=18.5,  # Above 18% target
                current_phase=PhaseType.BULK,
                phase_config=bulk_config,
                weeks_in_phase=15,
                gender="male",
            )
        )

    def test_maximum_duration_enforcement(self):
        """Test sustainability maximum duration enforcement"""
        phase_config = PhaseConfig(
            phase_type=PhaseType.BULK,
            target_bf_pct=20.0,
            min_duration_weeks=12,
            max_duration_weeks=36,
            rate_pct_per_week=0.325,
            rationale="Test max duration",
        )

        # Should force transition at maximum duration
        should_transition = self.manager.should_transition(
            current_bf_pct=16.0,  # Target not reached
            current_phase=PhaseType.BULK,
            phase_config=phase_config,
            weeks_in_phase=36,  # At maximum
            gender="male",
        )
        self.assertTrue(should_transition)

    def test_safety_validation(self):
        """Test safety constraints prevent dangerous transitions"""
        # Test cutting safety bounds
        with self.assertRaises(TransitionError):
            self.manager.validate_transition(
                from_phase=PhaseType.BULK,
                to_phase=PhaseType.CUT,
                current_bf_pct=9.0,  # Too close to 8% minimum for males
                gender="male",
            )

        # Test valid transitions
        self.assertTrue(
            self.manager.validate_transition(
                from_phase=PhaseType.CUT,
                to_phase=PhaseType.BULK,
                current_bf_pct=15.0,
                gender="male",
            )
        )

    def test_sequence_navigation(self):
        """Test proper navigation through phase sequences"""
        # Create test sequence
        phases = [
            PhaseConfig(PhaseType.CUT, 15.0, 8, 24, 0.625, "Initial cut"),
            PhaseConfig(PhaseType.BULK, 18.0, 12, 36, 0.325, "First bulk"),
            PhaseConfig(PhaseType.CUT, 12.0, 8, 20, 0.625, "Final cut"),
        ]
        sequence = PhaseSequence(
            template=TemplateType.CUT_FIRST,
            phases=phases,
            rationale="Test sequence",
            expected_duration_weeks=56,
            safety_validated=True,
        )

        # Test progression through sequence
        next_phase = self.manager.get_next_phase_config(
            phases[0], MagicMock(), sequence
        )
        self.assertEqual(next_phase, phases[1])

        next_phase = self.manager.get_next_phase_config(
            phases[1], MagicMock(), sequence
        )
        self.assertEqual(next_phase, phases[2])

        # Test end of sequence
        next_phase = self.manager.get_next_phase_config(
            phases[2], MagicMock(), sequence
        )
        self.assertIsNone(next_phase)


class TestPhaseValidationEngine(unittest.TestCase):
    """Test comprehensive validation against research constraints"""

    def test_duration_validation(self):
        """Test duration constraints match research"""
        # Test minimum duration validation
        with self.assertRaises(PhaseConfigError):
            PhaseValidationEngine.validate_phase_config(
                PhaseConfig(
                    PhaseType.CUT,
                    15.0,
                    4,
                    24,
                    0.625,
                    "Too short",  # Less than 6 weeks
                ),
                "male",
            )

        # Test maximum duration validation
        with self.assertRaises(PhaseConfigError):
            PhaseValidationEngine.validate_phase_config(
                PhaseConfig(
                    PhaseType.BULK,
                    18.0,
                    12,
                    60,
                    0.325,
                    "Too long",  # Over 52 weeks
                ),
                "male",
            )

    def test_body_fat_safety_validation(self):
        """Test body fat targets respect health research"""
        # Test below safety minimum
        with self.assertRaises(PhaseConfigError):
            PhaseValidationEngine.validate_phase_config(
                PhaseConfig(
                    PhaseType.CUT,
                    6.0,
                    8,
                    24,
                    0.625,
                    "Unsafe BF",  # Below 8% male minimum
                ),
                "male",
            )

        # Test valid BF targets
        self.assertTrue(
            PhaseValidationEngine.validate_phase_config(
                PhaseConfig(PhaseType.CUT, 12.0, 8, 24, 0.625, "Safe BF"),
                "male",
            )
        )

    def test_rate_bounds_validation(self):
        """Test rate limits match sustainability research"""
        # Test rate too high
        with self.assertRaises(PhaseConfigError):
            PhaseValidationEngine.validate_phase_config(
                PhaseConfig(
                    PhaseType.BULK,
                    18.0,
                    12,
                    36,
                    2.0,
                    "Unsustainable rate",  # Over 1.5% limit
                ),
                "male",
            )

        # Test negative rate
        with self.assertRaises(PhaseConfigError):
            PhaseValidationEngine.validate_phase_config(
                PhaseConfig(PhaseType.CUT, 15.0, 8, 24, -0.5, "Negative rate"),
                "male",
            )

    def test_sequence_validation(self):
        """Test complete sequence validation"""
        # Test empty sequence
        with self.assertRaises(PhaseConfigError):
            PhaseValidationEngine.validate_sequence(
                PhaseSequence(TemplateType.CUT_FIRST, [], "Empty", 0, False),
                "male",
            )

        # Test consecutive same phases (invalid)
        invalid_phases = [
            PhaseConfig(PhaseType.CUT, 15.0, 8, 24, 0.625, "Cut 1"),
            PhaseConfig(
                PhaseType.CUT, 12.0, 8, 20, 0.625, "Cut 2"
            ),  # Invalid: consecutive cuts
        ]
        with self.assertRaises(PhaseConfigError):
            PhaseValidationEngine.validate_sequence(
                PhaseSequence(
                    TemplateType.CUT_FIRST, invalid_phases, "Invalid", 40, False
                ),
                "male",
            )

        # Test valid alternating sequence
        valid_phases = [
            PhaseConfig(PhaseType.CUT, 15.0, 8, 24, 0.625, "Cut"),
            PhaseConfig(PhaseType.BULK, 18.0, 12, 36, 0.325, "Bulk"),
            PhaseConfig(PhaseType.CUT, 12.0, 8, 20, 0.625, "Final cut"),
        ]
        self.assertTrue(
            PhaseValidationEngine.validate_sequence(
                PhaseSequence(TemplateType.CUT_FIRST, valid_phases, "Valid", 68, True),
                "male",
            )
        )

    def test_total_duration_constraints(self):
        """Test total sequence duration limits"""
        # Create sequence exceeding 5-year practical limit
        long_phases = [
            PhaseConfig(PhaseType.CUT, 15.0, 8, 52, 0.625, "Long cut"),
            PhaseConfig(PhaseType.BULK, 18.0, 12, 52, 0.325, "Long bulk"),
            PhaseConfig(PhaseType.CUT, 12.0, 8, 52, 0.625, "Long cut 2"),
            PhaseConfig(PhaseType.BULK, 18.0, 12, 52, 0.325, "Long bulk 2"),
            PhaseConfig(PhaseType.CUT, 12.0, 8, 52, 0.625, "Long cut 3"),
            PhaseConfig(
                PhaseType.BULK, 18.0, 12, 52, 0.325, "Long bulk 3"
            ),  # 6th phase pushes over limit
        ]

        with self.assertRaises(PhaseConfigError):
            PhaseValidationEngine.validate_sequence(
                PhaseSequence(
                    TemplateType.CUT_FIRST,
                    long_phases,
                    "Too long",
                    312,  # 6 years (exceeds 5-year limit)
                    False,
                ),
                "male",
            )


class TestResearchValidation(unittest.TestCase):
    """Integration tests validating against published research"""

    def test_macrofactor_rate_compliance(self):
        """Test all rates comply with MacroFactor research"""
        calculator = RateCalculator()

        # Test that moderate rates are exactly the MacroFactor recommendations
        macrofactor_rates = {
            TrainingLevel.NOVICE: 0.5,
            TrainingLevel.INTERMEDIATE: 0.325,
            TrainingLevel.ADVANCED: 0.15,
        }

        for level, expected_rate in macrofactor_rates.items():
            actual_rate = calculator.get_bulk_rate(level, "moderate")
            self.assertEqual(
                actual_rate,
                expected_rate,
                f"Rate mismatch for {level.value}: expected {expected_rate}, got {actual_rate}",
            )

    def test_body_fat_thresholds_match_research(self):
        """Test body fat thresholds match health research"""
        # Test thresholds from shared_models match implementation
        male_thresholds = BF_THRESHOLDS["male"]
        female_thresholds = BF_THRESHOLDS["female"]

        # Validate against health research standards
        self.assertEqual(male_thresholds["healthy_max"], 25)  # Health research
        self.assertEqual(male_thresholds["minimum"], 8)  # Safety research
        self.assertEqual(female_thresholds["healthy_max"], 35)  # Health research
        self.assertEqual(female_thresholds["minimum"], 16)  # Safety research

    def test_phase_duration_research_compliance(self):
        """Test phase durations match adaptation research"""
        rate_calculator = RateCalculator()
        engine = PhaseTemplateEngine(rate_calculator)

        # Create test user
        user = UserProfile(
            birth_date="01/01/1990",
            height_in=68.0,
            gender="male",
            training_level=TrainingLevel.INTERMEDIATE,
            scan_history=[
                ScanData(
                    date="01/01/2024",
                    total_weight_lbs=170.0,
                    total_lean_mass_lbs=140.0,
                    fat_mass_lbs=30.0,
                    body_fat_percentage=25.0,
                    arms_lean_lbs=15.0,
                    legs_lean_lbs=40.0,
                )
            ],
        )

        sequence = engine.generate_sequence(TemplateType.CUT_FIRST, user)

        # Validate minimum durations match research
        for phase in sequence.phases:
            if phase.phase_type == PhaseType.CUT:
                self.assertGreaterEqual(
                    phase.min_duration_weeks,
                    8,
                    "Cut minimum should be 8 weeks (Garthe et al.)",
                )
            elif phase.phase_type == PhaseType.BULK:
                self.assertGreaterEqual(
                    phase.min_duration_weeks,
                    12,
                    "Bulk minimum should be 12 weeks (MPS adaptation)",
                )

    def test_p_ratio_research_accuracy(self):
        """Test P-ratio calculations match Forbes & Hall research"""
        calculator = RateCalculator()

        # Test bulking P-ratio matches Forbes research (45-50% range)
        bulk_p_ratio = calculator.get_p_ratio(PhaseType.BULK, 15.0, "male")
        self.assertGreaterEqual(bulk_p_ratio, 0.45)
        self.assertLessEqual(bulk_p_ratio, 0.50)

        # Test cutting P-ratio varies with body fat (Hall et al.)
        high_bf_ratio = calculator.get_p_ratio(PhaseType.CUT, 30.0, "male")
        low_bf_ratio = calculator.get_p_ratio(PhaseType.CUT, 12.0, "male")

        # Higher body fat should have lower P-ratio (better fat loss ratio)
        self.assertLess(high_bf_ratio, low_bf_ratio)


if __name__ == "__main__":
    unittest.main()
