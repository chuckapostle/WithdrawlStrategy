#!/usr/bin/env python3
"""
Retirement Withdrawal Strategy Planner v4
Interactive Streamlit App for comparing withdrawal strategies with:
- Roth conversion optimization
- Two configurable annuity streams (with start/end dates)
- Heir tax efficiency analysis
- Real dollar (inflation-adjusted) analysis

Strategies:
- S1: Tax-Optimized (401K first to fill 12% bracket)
- S2: After-Tax First
- S2 + Roth Conv: S2 with Roth conversions to fill 12% bracket

Run with: streamlit run retirement_planner.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Dict, List, Tuple
import io

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Retirement Withdrawal Planner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class SimulationParams:
    # Account Balances
    initial_401k: float
    initial_roth: float
    initial_after_tax: float
    
    # Income Sources
    social_security: float
    ss_start_age: int
    pension: float
    pension_start_age: int
    rental_income: float
    rental_start_age: int
    
    # Annuity 1
    annuity1_amount: float
    annuity1_start_age: int
    annuity1_end_age: int
    annuity1_name: str
    
    # Annuity 2
    annuity2_amount: float
    annuity2_start_age: int
    annuity2_end_age: int
    annuity2_name: str
    
    # Assumptions
    inflation_rate: float
    real_return_60_40: float
    fixed_income_return: float
    
    # Expenses
    initial_expenses: float
    
    # Tax Parameters
    tax_bracket_12_top: float
    standard_deduction: float
    
    # Heir Tax Parameters
    heir_federal_rate: float
    heir_state_rate: float
    after_tax_cost_basis_pct: float
    
    # Age Range
    start_age: int
    end_age: int
    
    # Monte Carlo
    run_monte_carlo: bool
    num_simulations: int
    return_std_dev: float

# =============================================================================
# RMD TABLE
# =============================================================================
RMD_DIVISORS = {
    72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0,
    79: 21.1, 80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0,
    86: 15.2, 87: 14.4, 88: 13.7, 89: 13.0, 90: 12.2, 91: 11.5, 92: 10.8,
    93: 10.1, 94: 9.5, 95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4
}

RMD_START_AGE = 75

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def nominal_to_real(nominal_value: float, years_from_start: int, inflation_rate: float) -> float:
    """Convert nominal dollars to real (today's) dollars"""
    return nominal_value / ((1 + inflation_rate) ** years_from_start)

def real_to_nominal(real_value: float, years_from_start: int, inflation_rate: float) -> float:
    """Convert real (today's) dollars to nominal dollars"""
    return real_value * ((1 + inflation_rate) ** years_from_start)

# =============================================================================
# TAX FUNCTIONS
# =============================================================================
def calculate_ss_taxable(ss_income: float, other_income: float) -> float:
    """Calculate taxable portion of Social Security"""
    if ss_income == 0:
        return 0
    provisional = other_income + (ss_income * 0.5)
    if provisional > 44_000:
        return ss_income * 0.85
    elif provisional > 32_000:
        return ss_income * 0.50
    return 0

def calculate_federal_tax(taxable_income: float, bracket_12_top: float) -> float:
    """Calculate federal tax using MFJ brackets"""
    if taxable_income <= 0:
        return 0
    
    scale = bracket_12_top / 97_000
    bracket_10_top = 23_200 * scale
    bracket_22_top = 206_700 * scale
    
    tax = 0
    remaining = taxable_income
    
    if remaining > 0:
        amt = min(remaining, bracket_10_top)
        tax += amt * 0.10
        remaining -= amt
    
    if remaining > 0:
        amt = min(remaining, bracket_12_top - bracket_10_top)
        tax += amt * 0.12
        remaining -= amt
    
    if remaining > 0:
        amt = min(remaining, bracket_22_top - bracket_12_top)
        tax += amt * 0.22
        remaining -= amt
    
    if remaining > 0:
        tax += remaining * 0.24
    
    return tax

def calculate_rmd(age: int, balance: float) -> float:
    """Calculate Required Minimum Distribution"""
    if age < RMD_START_AGE or balance <= 0:
        return 0
    divisor = RMD_DIVISORS.get(age, 6.0)
    return balance / divisor

def calculate_heir_after_tax_value(
    bal_401k: float,
    bal_roth: float, 
    bal_after_tax: float,
    heir_federal_rate: float,
    heir_state_rate: float,
    cost_basis_pct: float
) -> Dict:
    """Calculate the after-tax value to heirs for each account type."""
    heir_combined_rate = heir_federal_rate + heir_state_rate
    
    tax_on_401k = bal_401k * heir_combined_rate
    heir_value_401k = bal_401k - tax_on_401k
    
    tax_on_roth = 0
    heir_value_roth = bal_roth
    
    heir_ltcg_rate = 0.15
    potential_gain = bal_after_tax * 0.10  
    tax_on_after_tax = potential_gain * heir_ltcg_rate
    heir_value_after_tax = bal_after_tax - tax_on_after_tax
    
    total_pretax = bal_401k + bal_roth + bal_after_tax
    total_after_tax = heir_value_401k + heir_value_roth + heir_value_after_tax
    total_tax = tax_on_401k + tax_on_roth + tax_on_after_tax
    
    return {
        'bal_401k': bal_401k,
        'bal_roth': bal_roth,
        'bal_after_tax': bal_after_tax,
        'tax_on_401k': tax_on_401k,
        'tax_on_roth': tax_on_roth,
        'tax_on_after_tax': tax_on_after_tax,
        'heir_value_401k': heir_value_401k,
        'heir_value_roth': heir_value_roth,
        'heir_value_after_tax': heir_value_after_tax,
        'total_pretax': total_pretax,
        'total_after_tax_to_heirs': total_after_tax,
        'total_heir_tax': total_tax,
        'effective_heir_tax_rate': total_tax / total_pretax if total_pretax > 0 else 0
    }

# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================
def simulate_year(
    age: int,
    balances: Tuple[float, float, float],
    params: SimulationParams,
    strategy: str,
    year_return_60_40: float = None,
    year_return_fixed: float = None
) -> Dict:
    """Simulate one year of retirement"""
    bal_401k, bal_roth, bal_after_tax = balances
    
    if year_return_60_40 is None:
        nominal_return_60_40 = params.real_return_60_40 + params.inflation_rate
    else:
        nominal_return_60_40 = year_return_60_40
    
    if year_return_fixed is None:
        nominal_return_fixed = params.fixed_income_return
    else:
        nominal_return_fixed = year_return_fixed
    
    year_offset = age - params.start_age
    inflation_mult = (1 + params.inflation_rate) ** year_offset
    
    expenses = params.initial_expenses * inflation_mult
    bracket_12 = params.tax_bracket_12_top * inflation_mult
    std_ded = params.standard_deduction * inflation_mult
    
    ss_income = params.social_security if age >= params.ss_start_age else 0
    pension = params.pension if age >= params.pension_start_age else 0
    rental = (params.rental_income * inflation_mult) if age >= params.rental_start_age else 0
    
    # Annuity 1 (fixed nominal amount during active period)
    if age >= params.annuity1_start_age and age <= params.annuity1_end_age:
        annuity1 = params.annuity1_amount
    else:
        annuity1 = 0
    
    # Annuity 2 (fixed nominal amount during active period)
    if age >= params.annuity2_start_age and age <= params.annuity2_end_age:
        annuity2 = params.annuity2_amount
    else:
        annuity2 = 0
    
    total_other_income = ss_income + pension + rental + annuity1 + annuity2
    rmd = calculate_rmd(age, bal_401k)
    gap = max(0, expenses - total_other_income)
    
    ss_taxable = calculate_ss_taxable(ss_income, pension + rental + annuity1 + annuity2)
    base_ordinary = ss_taxable + pension + rental + annuity1 + annuity2
    agi_threshold_12 = std_ded + bracket_12
    
    if strategy == 's1':
        room_in_12 = max(0, agi_threshold_12 - base_ordinary)
        draw_401k = max(rmd, min(gap, room_in_12))
        draw_401k = min(draw_401k, bal_401k)
        
        gap_remaining = gap - draw_401k
        draw_after_tax = min(gap_remaining, bal_after_tax) if gap_remaining > 0 else 0
        gap_remaining -= draw_after_tax
        draw_roth = min(gap_remaining, bal_roth) if gap_remaining > 0 else 0
        roth_conversion = 0
        
    elif strategy == 's2':
        draw_after_tax = min(gap, bal_after_tax) if gap > 0 else 0
        gap_remaining = gap - draw_after_tax
        
        draw_401k = max(rmd, 0)
        if gap_remaining > 0:
            additional = min(gap_remaining, bal_401k - draw_401k)
            draw_401k += additional
            gap_remaining -= additional
        draw_401k = min(draw_401k, bal_401k)
        
        draw_roth = min(gap_remaining, bal_roth) if gap_remaining > 0 else 0
        roth_conversion = 0
        
    else:  # s2_roth
        draw_after_tax = min(gap, bal_after_tax) if gap > 0 else 0
        gap_remaining = gap - draw_after_tax
        
        draw_401k = max(rmd, 0)
        if gap_remaining > 0:
            additional = min(gap_remaining, bal_401k - draw_401k)
            draw_401k += additional
            gap_remaining -= additional
        draw_401k = min(draw_401k, bal_401k)
        
        draw_roth = min(gap_remaining, bal_roth) if gap_remaining > 0 else 0
        
        total_ordinary_after_draw = base_ordinary + draw_401k
        remaining_room = max(0, agi_threshold_12 - total_ordinary_after_draw)
        roth_conversion = min(remaining_room, bal_401k - draw_401k)
        roth_conversion = max(0, roth_conversion)
    
    total_401k_outflow = draw_401k + roth_conversion
    total_ordinary = base_ordinary + total_401k_outflow
    taxable_income = max(0, total_ordinary - std_ded)
    federal_tax = calculate_federal_tax(taxable_income, bracket_12)
    
    cap_gains = draw_after_tax * 0.5 if draw_after_tax > 0 else 0
    cap_gains_tax = cap_gains * 0.15
    total_tax = federal_tax + cap_gains_tax
    
    conversion_tax = roth_conversion * 0.12
    
    new_401k = (bal_401k - total_401k_outflow) * (1 + nominal_return_60_40)
    new_roth = (bal_roth - draw_roth + roth_conversion) * (1 + nominal_return_60_40)
    new_after_tax = (bal_after_tax - draw_after_tax) * (1 + nominal_return_fixed)
    
    total_balance = new_401k + new_roth + new_after_tax
    
    # Calculate real (today's dollar) values
    real_multiplier = 1 / inflation_mult
    
    return {
        'age': age,
        'year_offset': year_offset,
        # Nominal values
        'expenses': expenses,
        'ss_income': ss_income,
        'pension': pension,
        'rental': rental,
        'annuity1': annuity1,
        'annuity2': annuity2,
        'other_income': total_other_income,
        'draw_401k': draw_401k,
        'draw_after_tax': draw_after_tax,
        'draw_roth': draw_roth,
        'roth_conversion': roth_conversion,
        'conversion_tax': conversion_tax,
        'rmd_required': rmd,
        'taxable_income': taxable_income,
        'federal_tax': federal_tax,
        'cap_gains_tax': cap_gains_tax,
        'total_tax': total_tax,
        'effective_tax_rate': total_tax / expenses if expenses > 0 else 0,
        'bal_401k': new_401k,
        'bal_roth': new_roth,
        'bal_after_tax': new_after_tax,
        'total_balance': total_balance,
        # Real (today's dollar) values
        'real_expenses': expenses * real_multiplier,
        'real_other_income': total_other_income * real_multiplier,
        'real_annuity1': annuity1 * real_multiplier,
        'real_annuity2': annuity2 * real_multiplier,
        'real_total_tax': total_tax * real_multiplier,
        'real_bal_401k': new_401k * real_multiplier,
        'real_bal_roth': new_roth * real_multiplier,
        'real_bal_after_tax': new_after_tax * real_multiplier,
        'real_total_balance': total_balance * real_multiplier,
        'inflation_multiplier': inflation_mult,
        'new_balances': (new_401k, new_roth, new_after_tax)
    }

def run_simulation(params: SimulationParams, strategy: str) -> List[Dict]:
    """Run full deterministic simulation"""
    balances = (params.initial_401k, params.initial_roth, params.initial_after_tax)
    results = []
    cum_taxes = 0
    cum_taxes_real = 0
    cum_conversions = 0
    cum_conversion_tax = 0
    
    for age in range(params.start_age, params.end_age + 1):
        year_result = simulate_year(age, balances, params, strategy)
        cum_taxes += year_result['total_tax']
        cum_taxes_real += year_result['real_total_tax']
        cum_conversions += year_result['roth_conversion']
        cum_conversion_tax += year_result['conversion_tax']
        year_result['cum_taxes'] = cum_taxes
        year_result['cum_taxes_real'] = cum_taxes_real
        year_result['cum_conversions'] = cum_conversions
        year_result['cum_conversion_tax'] = cum_conversion_tax
        results.append(year_result)
        balances = year_result['new_balances']
    
    return results

def run_monte_carlo(params: SimulationParams, strategy: str) -> Dict:
    """Run Monte Carlo simulation with random returns"""
    all_final_balances = []
    all_final_balances_real = []
    all_cum_taxes = []
    all_cum_taxes_real = []
    all_trajectories = []
    all_trajectories_real = []
    all_final_401k = []
    all_final_roth = []
    all_final_after_tax = []
    
    total_years = params.end_age - params.start_age + 1
    final_inflation_mult = (1 + params.inflation_rate) ** total_years
    
    for sim in range(params.num_simulations):
        balances = (params.initial_401k, params.initial_roth, params.initial_after_tax)
        trajectory = [sum(balances)]
        trajectory_real = [sum(balances)]  # At start, real = nominal
        cum_taxes = 0
        cum_taxes_real = 0
        
        for age in range(params.start_age, params.end_age + 1):
            year_offset = age - params.start_age
            real_return = np.random.normal(params.real_return_60_40, params.return_std_dev)
            nominal_return_60_40 = real_return + params.inflation_rate
            nominal_return_fixed = params.fixed_income_return + np.random.normal(0, params.return_std_dev * 0.3)
            
            year_result = simulate_year(
                age, balances, params, strategy,
                year_return_60_40=nominal_return_60_40,
                year_return_fixed=nominal_return_fixed
            )
            cum_taxes += year_result['total_tax']
            cum_taxes_real += year_result['real_total_tax']
            balances = year_result['new_balances']
            trajectory.append(sum(balances))
            trajectory_real.append(year_result['real_total_balance'])
        
        all_final_balances.append(sum(balances))
        all_final_balances_real.append(sum(balances) / final_inflation_mult)
        all_cum_taxes.append(cum_taxes)
        all_cum_taxes_real.append(cum_taxes_real)
        all_trajectories.append(trajectory)
        all_trajectories_real.append(trajectory_real)
        all_final_401k.append(balances[0])
        all_final_roth.append(balances[1])
        all_final_after_tax.append(balances[2])
    
    return {
        'final_balances': np.array(all_final_balances),
        'final_balances_real': np.array(all_final_balances_real),
        'cum_taxes': np.array(all_cum_taxes),
        'cum_taxes_real': np.array(all_cum_taxes_real),
        'trajectories': np.array(all_trajectories),
        'trajectories_real': np.array(all_trajectories_real),
        'final_401k': np.array(all_final_401k),
        'final_roth': np.array(all_final_roth),
        'final_after_tax': np.array(all_final_after_tax),
        'median_final': np.median(all_final_balances),
        'median_final_real': np.median(all_final_balances_real),
        'p10_final': np.percentile(all_final_balances, 10),
        'p90_final': np.percentile(all_final_balances, 90),
        'p10_final_real': np.percentile(all_final_balances_real, 10),
        'p90_final_real': np.percentile(all_final_balances_real, 90),
        'median_taxes': np.median(all_cum_taxes),
        'median_taxes_real': np.median(all_cum_taxes_real),
    }

# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    st.title("📊 Retirement Withdrawal Strategy Planner")
    st.markdown("*Compare withdrawal strategies with real dollar analysis & heir tax efficiency*")
    
    # ==========================================================================
    # SIDEBAR - PARAMETERS
    # ==========================================================================
    with st.sidebar:
        st.header("⚙️ Parameters")
        
        # Account Balances
        st.subheader("💰 Account Balances")
        initial_401k = st.number_input("401K Balance ($)", value=1_500_000, step=50_000, format="%d")
        initial_roth = st.number_input("Roth Balance ($)", value=656_000, step=25_000, format="%d")
        initial_after_tax = st.number_input("After-Tax Balance ($)", value=740_000, step=25_000, format="%d")
        
        st.divider()
        
        # Income Sources
        st.subheader("📈 Income Sources")
        
        col1, col2 = st.columns(2)
        with col1:
            social_security = st.number_input("Social Security ($)", value=78_000, step=1_000, format="%d")
        with col2:
            ss_start_age = st.number_input("SS Start Age", value=63, min_value=62, max_value=70)
        
        col1, col2 = st.columns(2)
        with col1:
            pension = st.number_input("Pension ($)", value=22_000, step=1_000, format="%d")
        with col2:
            pension_start_age = st.number_input("Pension Start", value=63, min_value=55, max_value=75)
        
        col1, col2 = st.columns(2)
        with col1:
            rental_income = st.number_input("Rental Income ($)", value=25_000, step=1_000, format="%d")
        with col2:
            rental_start_age = st.number_input("Rental Start", value=63, min_value=55, max_value=75)
        
        st.divider()
        
        # Annuity 1
        st.subheader("💵 Annuity 1")
        annuity1_name = st.text_input("Name", value="Annuity 1", key="ann1_name")
        col1, col2 = st.columns(2)
        with col1:
            annuity1_amount = st.number_input("Annual Amount ($)", value=0, step=5_000, format="%d", key="ann1_amt")
        with col2:
            annuity1_start_age = st.number_input("Start Age", value=63, min_value=55, max_value=95, key="ann1_start")
        annuity1_end_age = st.number_input("End Age (inclusive)", value=72, min_value=55, max_value=100, key="ann1_end")
        
        if annuity1_amount > 0:
            duration1 = annuity1_end_age - annuity1_start_age + 1
            st.caption(f"📅 {annuity1_name}: Ages {annuity1_start_age}-{annuity1_end_age} ({duration1} years)")
        
        st.divider()
        
        # Annuity 2
        st.subheader("💵 Annuity 2")
        annuity2_name = st.text_input("Name", value="Annuity 2", key="ann2_name")
        col1, col2 = st.columns(2)
        with col1:
            annuity2_amount = st.number_input("Annual Amount ($)", value=0, step=5_000, format="%d", key="ann2_amt")
        with col2:
            annuity2_start_age = st.number_input("Start Age", value=70, min_value=55, max_value=95, key="ann2_start")
        annuity2_end_age = st.number_input("End Age (inclusive)", value=79, min_value=55, max_value=100, key="ann2_end")
        
        if annuity2_amount > 0:
            duration2 = annuity2_end_age - annuity2_start_age + 1
            st.caption(f"📅 {annuity2_name}: Ages {annuity2_start_age}-{annuity2_end_age} ({duration2} years)")
        
        st.divider()
        
        # Assumptions
        st.subheader("📊 Assumptions")
        initial_expenses = st.number_input("Annual Expenses ($)", value=150_000, step=5_000, format="%d")
        inflation_rate = st.slider("Inflation Rate (%)", 0.0, 5.0, 2.66, 0.1) / 100
        real_return_60_40 = st.slider("60/40 Real Return (%)", 0.0, 10.0, 6.0, 0.5) / 100
        fixed_income_return = st.slider("Fixed Income Return (%)", 0.0, 8.0, 5.0, 0.25) / 100
        
        st.divider()
        
        # Tax Parameters
        st.subheader("🏛️ Tax Parameters")
        tax_bracket_12_top = st.number_input("12% Bracket Top ($)", value=97_000, step=1_000, format="%d")
        standard_deduction = st.number_input("Standard Deduction ($)", value=32_500, step=500, format="%d")
        
        st.divider()
        
        # Heir Tax Parameters
        st.subheader("👨‍👩‍👧‍👦 Heir Tax Assumptions")
        heir_federal_rate = st.slider("Heir Federal Tax Rate (%)", 10, 37, 24, 1) / 100
        heir_state_rate = st.slider("Heir State Tax Rate (%)", 0, 13, 5, 1) / 100
        after_tax_cost_basis_pct = st.slider("After-Tax Cost Basis (%)", 30, 90, 50, 5) / 100
        
        st.caption(f"Combined heir tax rate: {(heir_federal_rate + heir_state_rate)*100:.0f}%")
        
        st.divider()
        
        # Age Range
        st.subheader("📅 Age Range")
        col1, col2 = st.columns(2)
        with col1:
            start_age = st.number_input("Start Age", value=63, min_value=55, max_value=75)
        with col2:
            end_age = st.number_input("End Age", value=89, min_value=75, max_value=100)
        
        st.divider()
        
        # Monte Carlo
        st.subheader("🎲 Monte Carlo")
        run_monte_carlo_sim = st.checkbox("Run Monte Carlo Simulation", value=False)
        if run_monte_carlo_sim:
            num_simulations = st.slider("Number of Simulations", 100, 5000, 1000, 100)
            return_std_dev = st.slider("Return Std Dev (%)", 5.0, 20.0, 12.0, 1.0) / 100
        else:
            num_simulations = 1000
            return_std_dev = 0.12
    
    # Build params object
    params = SimulationParams(
        initial_401k=initial_401k,
        initial_roth=initial_roth,
        initial_after_tax=initial_after_tax,
        social_security=social_security,
        ss_start_age=ss_start_age,
        pension=pension,
        pension_start_age=pension_start_age,
        rental_income=rental_income,
        rental_start_age=rental_start_age,
        annuity1_amount=annuity1_amount,
        annuity1_start_age=annuity1_start_age,
        annuity1_end_age=annuity1_end_age,
        annuity1_name=annuity1_name,
        annuity2_amount=annuity2_amount,
        annuity2_start_age=annuity2_start_age,
        annuity2_end_age=annuity2_end_age,
        annuity2_name=annuity2_name,
        inflation_rate=inflation_rate,
        real_return_60_40=real_return_60_40,
        fixed_income_return=fixed_income_return,
        initial_expenses=initial_expenses,
        tax_bracket_12_top=tax_bracket_12_top,
        standard_deduction=standard_deduction,
        heir_federal_rate=heir_federal_rate,
        heir_state_rate=heir_state_rate,
        after_tax_cost_basis_pct=after_tax_cost_basis_pct,
        start_age=start_age,
        end_age=end_age,
        run_monte_carlo=run_monte_carlo_sim,
        num_simulations=num_simulations,
        return_std_dev=return_std_dev
    )
    
    # ==========================================================================
    # RUN SIMULATIONS
    # ==========================================================================
    with st.spinner("Running simulations..."):
        results_s1 = run_simulation(params, 's1')
        results_s2 = run_simulation(params, 's2')
        results_s2_roth = run_simulation(params, 's2_roth')
        
        if run_monte_carlo_sim:
            mc_s1 = run_monte_carlo(params, 's1')
            mc_s2 = run_monte_carlo(params, 's2')
            mc_s2_roth = run_monte_carlo(params, 's2_roth')
    
    strategies = {
        'S1: Tax-Optimized': results_s1,
        'S2: After-Tax First': results_s2,
        'S2 + Roth Conv': results_s2_roth
    }
    
    # Calculate final inflation multiplier for real dollar conversions
    total_years = end_age - start_age
    final_inflation_mult = (1 + inflation_rate) ** total_years
    
    # ==========================================================================
    # SUMMARY METRICS
    # ==========================================================================
    st.header("📋 Summary Results")
    
    # Toggle for nominal vs real
    show_real = st.toggle("Show in Today's Dollars (Real)", value=False, 
                          help="Adjust all values for inflation to show purchasing power in today's dollars")
    
    # Create summary dataframe
    summary_data = []
    for name, results in strategies.items():
        final = results[-1]
        if show_real:
            summary_data.append({
                'Strategy': name,
                'Final Balance': final['real_total_balance'],
                'Final 401K': final['real_bal_401k'],
                'Final Roth': final['real_bal_roth'],
                'Final After-Tax': final['real_bal_after_tax'],
                'Total Taxes': final['cum_taxes_real'],
                'Total Conversions': final['cum_conversions'] / final['inflation_multiplier']
            })
        else:
            summary_data.append({
                'Strategy': name,
                'Final Balance': final['total_balance'],
                'Final 401K': final['bal_401k'],
                'Final Roth': final['bal_roth'],
                'Final After-Tax': final['bal_after_tax'],
                'Total Taxes': final['cum_taxes'],
                'Total Conversions': final['cum_conversions']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Final Balance', ascending=False)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    winner = summary_df.iloc[0]
    dollar_label = "Today's $" if show_real else "Nominal $"
    
    with col1:
        st.metric("🏆 Best Strategy", winner['Strategy'].split(':')[0] if ':' in winner['Strategy'] else winner['Strategy'])
    with col2:
        st.metric(f"Final Balance ({dollar_label})", f"${winner['Final Balance']:,.0f}")
    with col3:
        st.metric(f"Total Taxes ({dollar_label})", f"${winner['Total Taxes']:,.0f}")
    with col4:
        st.metric("Roth Converted", f"${winner['Total Conversions']:,.0f}")
    
    # Summary table
    st.subheader(f"Strategy Comparison ({dollar_label})")
    
    display_df = summary_df.copy()
    for col in ['Final Balance', 'Final 401K', 'Final Roth', 'Final After-Tax', 'Total Taxes', 'Total Conversions']:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # ==========================================================================
    # REAL DOLLAR ANALYSIS
    # ==========================================================================
    st.header("💵 Real Dollar (Inflation-Adjusted) Analysis")
    
    st.markdown(f"""
    *At {inflation_rate*100:.2f}% annual inflation over {total_years} years, 
    $1 today = ${final_inflation_mult:.2f} in nominal terms at age {end_age}*
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Nominal vs Real Final Balances")
        
        comparison_data = []
        for name, results in strategies.items():
            final = results[-1]
            comparison_data.append({
                'Strategy': name,
                'Nominal': final['total_balance'],
                'Real (Today\'s $)': final['real_total_balance'],
                'Purchasing Power Loss': final['total_balance'] - final['real_total_balance']
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Nominal $',
            x=comp_df['Strategy'],
            y=comp_df['Nominal'],
            marker_color='#1976D2',
            text=[f"${x/1e6:.1f}M" for x in comp_df['Nominal']],
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            name="Real (Today's $)",
            x=comp_df['Strategy'],
            y=comp_df['Real (Today\'s $)'],
            marker_color='#388E3C',
            text=[f"${x/1e6:.1f}M" for x in comp_df['Real (Today\'s $)']],
            textposition='outside'
        ))
        
        fig.update_layout(
            barmode='group',
            yaxis_tickformat="$,.0f",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Real Value Over Time (S1)")
        
        ages = [r['age'] for r in results_s1]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ages,
            y=[r['total_balance'] for r in results_s1],
            name='Nominal',
            line=dict(color='#1976D2', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=ages,
            y=[r['real_total_balance'] for r in results_s1],
            name="Real (Today's $)",
            line=dict(color='#388E3C', width=3)
        ))
        
        fig.update_layout(
            xaxis_title="Age",
            yaxis_title="Balance ($)",
            yaxis_tickformat="$,.0f",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Real dollar summary table
    st.subheader("Detailed Real Dollar Comparison")
    
    real_summary = []
    for name, results in strategies.items():
        final = results[-1]
        real_summary.append({
            'Strategy': name,
            'Nominal Final': f"${final['total_balance']:,.0f}",
            'Real Final': f"${final['real_total_balance']:,.0f}",
            'Nominal Taxes': f"${final['cum_taxes']:,.0f}",
            'Real Taxes': f"${final['cum_taxes_real']:,.0f}",
            'Real 401K': f"${final['real_bal_401k']:,.0f}",
            'Real Roth': f"${final['real_bal_roth']:,.0f}",
            'Real After-Tax': f"${final['real_bal_after_tax']:,.0f}"
        })
    
    st.dataframe(pd.DataFrame(real_summary), use_container_width=True, hide_index=True)
    
    # Annuity real value analysis
    if annuity1_amount > 0 or annuity2_amount > 0:
        st.subheader("📉 Annuity Purchasing Power Erosion")
        
        st.markdown("""
        Fixed annuities lose purchasing power over time as inflation erodes their real value.
        """)
        
        annuity_erosion = []
        for r in results_s1:
            if r['annuity1'] > 0 or r['annuity2'] > 0:
                annuity_erosion.append({
                    'Age': r['age'],
                    f'{annuity1_name} Nominal': r['annuity1'],
                    f'{annuity1_name} Real': r['real_annuity1'],
                    f'{annuity2_name} Nominal': r['annuity2'],
                    f'{annuity2_name} Real': r['real_annuity2'],
                })
        
        if annuity_erosion:
            erosion_df = pd.DataFrame(annuity_erosion)
            
            fig = go.Figure()
            
            if annuity1_amount > 0:
                fig.add_trace(go.Scatter(
                    x=erosion_df['Age'],
                    y=erosion_df[f'{annuity1_name} Nominal'],
                    name=f'{annuity1_name} (Nominal)',
                    line=dict(color='#7B1FA2', dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=erosion_df['Age'],
                    y=erosion_df[f'{annuity1_name} Real'],
                    name=f'{annuity1_name} (Real)',
                    line=dict(color='#7B1FA2', width=3)
                ))
            
            if annuity2_amount > 0:
                fig.add_trace(go.Scatter(
                    x=erosion_df['Age'],
                    y=erosion_df[f'{annuity2_name} Nominal'],
                    name=f'{annuity2_name} (Nominal)',
                    line=dict(color='#00897B', dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=erosion_df['Age'],
                    y=erosion_df[f'{annuity2_name} Real'],
                    name=f'{annuity2_name} (Real)',
                    line=dict(color='#00897B', width=3)
                ))
            
            fig.update_layout(
                title="Annuity Value: Nominal vs Real",
                xaxis_title="Age",
                yaxis_title="Annual Payment ($)",
                yaxis_tickformat="$,.0f",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show erosion stats
            if annuity1_amount > 0:
                first_real = [r['real_annuity1'] for r in results_s1 if r['annuity1'] > 0][0]
                last_real = [r['real_annuity1'] for r in results_s1 if r['annuity1'] > 0][-1]
                erosion_pct = (1 - last_real / first_real) * 100
                st.metric(f"{annuity1_name} Purchasing Power Lost", 
                         f"{erosion_pct:.1f}%",
                         delta=f"-${annuity1_amount - last_real:,.0f}/yr in real terms")
            
            if annuity2_amount > 0:
                first_real = [r['real_annuity2'] for r in results_s1 if r['annuity2'] > 0][0]
                last_real = [r['real_annuity2'] for r in results_s1 if r['annuity2'] > 0][-1]
                erosion_pct = (1 - last_real / first_real) * 100
                st.metric(f"{annuity2_name} Purchasing Power Lost",
                         f"{erosion_pct:.1f}%",
                         delta=f"-${annuity2_amount - last_real:,.0f}/yr in real terms")
    
    # ==========================================================================
    # HEIR TAX EFFICIENCY ANALYSIS
    # ==========================================================================
    st.header("👨‍👩‍👧‍👦 Heir Tax Efficiency Analysis")
    
    st.markdown(f"""
    *Assuming heirs are in the **{(heir_federal_rate)*100:.0f}% federal** + **{(heir_state_rate)*100:.0f}% state** tax bracket 
    (combined **{(heir_federal_rate + heir_state_rate)*100:.0f}%**)*
    """)
    
    # Calculate heir values for each strategy (in both nominal and real)
    heir_analysis = []
    for name, results in strategies.items():
        final = results[-1]
        heir_calc = calculate_heir_after_tax_value(
            final['bal_401k'],
            final['bal_roth'],
            final['bal_after_tax'],
            heir_federal_rate,
            heir_state_rate,
            after_tax_cost_basis_pct
        )
        heir_calc['strategy'] = name
        heir_calc['lifetime_taxes_paid'] = final['cum_taxes']
        heir_calc['total_conversion_tax'] = final.get('cum_conversion_tax', 0)
        # Add real values
        heir_calc['real_total_after_tax_to_heirs'] = heir_calc['total_after_tax_to_heirs'] / final_inflation_mult
        heir_calc['real_total_pretax'] = heir_calc['total_pretax'] / final_inflation_mult
        heir_analysis.append(heir_calc)
    
    heir_df = pd.DataFrame(heir_analysis)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pre-Tax vs After-Tax to Heirs")
        
        heir_compare = []
        for _, row in heir_df.iterrows():
            if show_real:
                heir_compare.append({
                    'Strategy': row['strategy'],
                    'Pre-Tax Legacy': row['real_total_pretax'],
                    'Tax Heirs Pay': row['total_heir_tax'] / final_inflation_mult,
                    'After-Tax to Heirs': row['real_total_after_tax_to_heirs'],
                    'Heir Tax Rate': row['effective_heir_tax_rate']
                })
            else:
                heir_compare.append({
                    'Strategy': row['strategy'],
                    'Pre-Tax Legacy': row['total_pretax'],
                    'Tax Heirs Pay': row['total_heir_tax'],
                    'After-Tax to Heirs': row['total_after_tax_to_heirs'],
                    'Heir Tax Rate': row['effective_heir_tax_rate']
                })
        
        heir_compare_df = pd.DataFrame(heir_compare).sort_values('After-Tax to Heirs', ascending=False)
        
        display_heir = heir_compare_df.copy()
        for col in ['Pre-Tax Legacy', 'Tax Heirs Pay', 'After-Tax to Heirs']:
            display_heir[col] = display_heir[col].apply(lambda x: f"${x:,.0f}")
        display_heir['Heir Tax Rate'] = display_heir['Heir Tax Rate'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(display_heir, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Account Breakdown at Death")
        
        fig = go.Figure()
        
        strategies_list = heir_df['strategy'].tolist()
        
        fig.add_trace(go.Bar(
            name='401K (taxable to heirs)',
            x=strategies_list,
            y=heir_df['bal_401k'] if not show_real else heir_df['bal_401k'] / final_inflation_mult,
            marker_color='#EF5350',
            text=[f"${x/1e6:.1f}M" for x in (heir_df['bal_401k'] if not show_real else heir_df['bal_401k'] / final_inflation_mult)],
            textposition='inside'
        ))
        fig.add_trace(go.Bar(
            name='Roth (tax-free to heirs)',
            x=strategies_list,
            y=heir_df['bal_roth'] if not show_real else heir_df['bal_roth'] / final_inflation_mult,
            marker_color='#66BB6A',
            text=[f"${x/1e6:.1f}M" for x in (heir_df['bal_roth'] if not show_real else heir_df['bal_roth'] / final_inflation_mult)],
            textposition='inside'
        ))
        fig.add_trace(go.Bar(
            name='After-Tax (step-up basis)',
            x=strategies_list,
            y=heir_df['bal_after_tax'] if not show_real else heir_df['bal_after_tax'] / final_inflation_mult,
            marker_color='#FFA726',
            text=[f"${x/1e6:.1f}M" for x in (heir_df['bal_after_tax'] if not show_real else heir_df['bal_after_tax'] / final_inflation_mult)],
            textposition='inside'
        ))
        
        fig.update_layout(
            barmode='stack',
            title=f"Legacy by Account Type ({dollar_label})",
            yaxis_title="Balance ($)",
            yaxis_tickformat="$,.0f",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # ROTH CONVERSION ROI ANALYSIS
    # ==========================================================================
    st.subheader("🔄 Was the Roth Conversion Worth It?")
    
    s2_final = results_s2[-1]
    s2r_final = results_s2_roth[-1]
    
    s2_heir = calculate_heir_after_tax_value(
        s2_final['bal_401k'], s2_final['bal_roth'], s2_final['bal_after_tax'],
        heir_federal_rate, heir_state_rate, after_tax_cost_basis_pct
    )
    s2r_heir = calculate_heir_after_tax_value(
        s2r_final['bal_401k'], s2r_final['bal_roth'], s2r_final['bal_after_tax'],
        heir_federal_rate, heir_state_rate, after_tax_cost_basis_pct
    )
    
    total_converted = s2r_final['cum_conversions']
    conversion_tax_paid = s2r_final.get('cum_conversion_tax', total_converted * 0.12)
    
    extra_taxes_during_life = s2r_final['cum_taxes'] - s2_final['cum_taxes']
    heir_tax_saved = s2_heir['total_heir_tax'] - s2r_heir['total_heir_tax']
    extra_to_heirs = s2r_heir['total_after_tax_to_heirs'] - s2_heir['total_after_tax_to_heirs']
    
    # Real dollar versions
    extra_taxes_real = s2r_final['cum_taxes_real'] - s2_final['cum_taxes_real']
    extra_to_heirs_real = extra_to_heirs / final_inflation_mult
    heir_tax_saved_real = heir_tax_saved / final_inflation_mult
    
    net_benefit = extra_to_heirs - extra_taxes_during_life
    net_benefit_real = extra_to_heirs_real - extra_taxes_real
    roi = (extra_to_heirs / extra_taxes_during_life - 1) * 100 if extra_taxes_during_life > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if show_real:
            st.metric("Total Converted to Roth", f"${total_converted/final_inflation_mult:,.0f}")
            st.metric("Extra Tax Paid (Your Lifetime)", f"${extra_taxes_real:,.0f}")
        else:
            st.metric("Total Converted to Roth", f"${total_converted:,.0f}")
            st.metric("Extra Tax Paid (Your Lifetime)", f"${extra_taxes_during_life:,.0f}")
    
    with col2:
        if show_real:
            st.metric("Tax Your Heirs Avoid", f"${heir_tax_saved_real:,.0f}", delta="saved")
            st.metric("Extra to Heirs (After-Tax)", f"${extra_to_heirs_real:,.0f}",
                      delta=f"+${extra_to_heirs_real:,.0f}" if extra_to_heirs_real > 0 else f"-${abs(extra_to_heirs_real):,.0f}")
        else:
            st.metric("Tax Your Heirs Avoid", f"${heir_tax_saved:,.0f}", delta="saved")
            st.metric("Extra to Heirs (After-Tax)", f"${extra_to_heirs:,.0f}",
                      delta=f"+${extra_to_heirs:,.0f}" if extra_to_heirs > 0 else f"-${abs(extra_to_heirs):,.0f}")
    
    with col3:
        if show_real:
            st.metric("Net Benefit (Real $)", f"${net_benefit_real:,.0f}",
                      delta="Worth it! ✅" if net_benefit_real > 0 else "Not worth it ❌",
                      delta_color="normal" if net_benefit_real > 0 else "inverse")
        else:
            st.metric("Net Benefit (Nominal)", f"${net_benefit:,.0f}",
                      delta="Worth it! ✅" if net_benefit > 0 else "Not worth it ❌",
                      delta_color="normal" if net_benefit > 0 else "inverse")
        st.metric("ROI on Conversion Tax", f"{roi:.0f}%")
    
    if net_benefit > 0:
        st.success(f"""
        **✅ Roth conversions ARE worth it for your heirs!**
        
        You pay extra tax at ~12%, but heirs avoid tax at {(heir_federal_rate + heir_state_rate)*100:.0f}%.
        The {(heir_federal_rate + heir_state_rate)*100 - 12:.0f}% rate difference creates a **{roi:.0f}% return** on conversion taxes paid.
        """)
    else:
        st.warning(f"""
        **❌ Roth conversions may NOT be worth it in this scenario.**
        
        Check if heir tax rates are close to your conversion rate, or if the time horizon is short.
        """)
    
    # ==========================================================================
    # RECOMMENDATION
    # ==========================================================================
    st.header("🎯 Recommendation")
    
    s1_heir = calculate_heir_after_tax_value(
        results_s1[-1]['bal_401k'], results_s1[-1]['bal_roth'], results_s1[-1]['bal_after_tax'],
        heir_federal_rate, heir_state_rate, after_tax_cost_basis_pct
    )
    
    all_heir_values = {
        'S1: Tax-Optimized': s1_heir['total_after_tax_to_heirs'],
        'S2: After-Tax First': s2_heir['total_after_tax_to_heirs'],
        'S2 + Roth Conv': s2r_heir['total_after_tax_to_heirs']
    }
    
    if show_real:
        best_balance = max(strategies.items(), key=lambda x: x[1][-1]['real_total_balance'])[0]
        all_heir_values_display = {k: v/final_inflation_mult for k, v in all_heir_values.items()}
    else:
        best_balance = summary_df.iloc[0]['Strategy']
        all_heir_values_display = all_heir_values
    
    best_for_heirs = max(all_heir_values, key=all_heir_values.get)
    lowest_taxes = min(strategies.items(), key=lambda x: x[1][-1]['cum_taxes'])[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Best Total Balance:**\n\n{best_balance}")
    with col2:
        st.success(f"**Best for Heirs:**\n\n{best_for_heirs}\n\n${all_heir_values_display[best_for_heirs]:,.0f}")
    with col3:
        st.warning(f"**Lowest Lifetime Taxes:**\n\n{lowest_taxes}")
    
    # ==========================================================================
    # CHARTS
    # ==========================================================================
    st.header("📈 Visualizations")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Total Balance", "Income Sources", "Account Breakdown", "Taxes", "Roth Conversions"])
    
    colors = {
        'S1: Tax-Optimized': '#2E7D32',
        'S2: After-Tax First': '#E65100',
        'S2 + Roth Conv': '#7B1FA2'
    }
    
    with tab1:
        fig = go.Figure()
        
        for name, results in strategies.items():
            ages = [r['age'] for r in results]
            if show_real:
                balances = [r['real_total_balance'] for r in results]
            else:
                balances = [r['total_balance'] for r in results]
            fig.add_trace(go.Scatter(
                x=ages, y=balances, name=name,
                line=dict(color=colors[name], width=2)
            ))
        
        fig.add_vline(x=75, line_dash="dot", line_color="red", annotation_text="RMD Start")
        
        fig.update_layout(
            title=f"Total Portfolio Balance Over Time ({dollar_label})",
            xaxis_title="Age",
            yaxis_title="Balance ($)",
            yaxis_tickformat="$,.0f",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Income sources stacked bar
        income_data = []
        for r in results_s1:
            if show_real:
                income_data.append({
                    'Age': r['age'],
                    'Social Security': r['ss_income'] / r['inflation_multiplier'],
                    'Pension': r['pension'] / r['inflation_multiplier'],
                    'Rental': r['rental'] / r['inflation_multiplier'],
                    annuity1_name: r['real_annuity1'],
                    annuity2_name: r['real_annuity2'],
                })
            else:
                income_data.append({
                    'Age': r['age'],
                    'Social Security': r['ss_income'],
                    'Pension': r['pension'],
                    'Rental': r['rental'],
                    annuity1_name: r['annuity1'],
                    annuity2_name: r['annuity2'],
                })
        
        income_df = pd.DataFrame(income_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=income_df['Age'], y=income_df['Social Security'], name='Social Security', marker_color='#1976D2'))
        fig.add_trace(go.Bar(x=income_df['Age'], y=income_df['Pension'], name='Pension', marker_color='#388E3C'))
        fig.add_trace(go.Bar(x=income_df['Age'], y=income_df['Rental'], name='Rental', marker_color='#F57C00'))
        if annuity1_amount > 0:
            fig.add_trace(go.Bar(x=income_df['Age'], y=income_df[annuity1_name], name=annuity1_name, marker_color='#7B1FA2'))
        if annuity2_amount > 0:
            fig.add_trace(go.Bar(x=income_df['Age'], y=income_df[annuity2_name], name=annuity2_name, marker_color='#00897B'))
        
        fig.update_layout(
            barmode='stack',
            title=f"Annual Income by Source ({dollar_label})",
            xaxis_title="Age",
            yaxis_title="Income ($)",
            yaxis_tickformat="$,.0f",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        strategy_select = st.selectbox("Select Strategy", list(strategies.keys()))
        results = strategies[strategy_select]
        ages = [r['age'] for r in results]
        
        fig = go.Figure()
        if show_real:
            fig.add_trace(go.Scatter(x=ages, y=[r['real_bal_401k'] for r in results],
                name='401K', fill='tonexty', stackgroup='one', line=dict(color='#1976D2')))
            fig.add_trace(go.Scatter(x=ages, y=[r['real_bal_roth'] for r in results],
                name='Roth', fill='tonexty', stackgroup='one', line=dict(color='#388E3C')))
            fig.add_trace(go.Scatter(x=ages, y=[r['real_bal_after_tax'] for r in results],
                name='After-Tax', fill='tonexty', stackgroup='one', line=dict(color='#F57C00')))
        else:
            fig.add_trace(go.Scatter(x=ages, y=[r['bal_401k'] for r in results],
                name='401K', fill='tonexty', stackgroup='one', line=dict(color='#1976D2')))
            fig.add_trace(go.Scatter(x=ages, y=[r['bal_roth'] for r in results],
                name='Roth', fill='tonexty', stackgroup='one', line=dict(color='#388E3C')))
            fig.add_trace(go.Scatter(x=ages, y=[r['bal_after_tax'] for r in results],
                name='After-Tax', fill='tonexty', stackgroup='one', line=dict(color='#F57C00')))
        
        fig.update_layout(
            title=f"Account Breakdown: {strategy_select} ({dollar_label})",
            xaxis_title="Age",
            yaxis_title="Balance ($)",
            yaxis_tickformat="$,.0f",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        fig = go.Figure()
        
        for name, results in strategies.items():
            ages = [r['age'] for r in results]
            if show_real:
                taxes = [r['cum_taxes_real'] for r in results]
            else:
                taxes = [r['cum_taxes'] for r in results]
            fig.add_trace(go.Scatter(x=ages, y=taxes, name=name, line=dict(color=colors[name], width=2)))
        
        fig.update_layout(
            title=f"Cumulative Taxes Paid ({dollar_label})",
            xaxis_title="Age",
            yaxis_title="Cumulative Taxes ($)",
            yaxis_tickformat="$,.0f",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        fig = go.Figure()
        ages = [r['age'] for r in results_s2_roth]
        
        if show_real:
            conversions = [r['roth_conversion'] / r['inflation_multiplier'] for r in results_s2_roth]
        else:
            conversions = [r['roth_conversion'] for r in results_s2_roth]
        
        fig.add_trace(go.Bar(x=ages, y=conversions, name='S2 + Roth Conv', marker_color='#7B1FA2'))
        
        fig.update_layout(
            title=f"Annual Roth Conversions ({dollar_label})",
            xaxis_title="Age",
            yaxis_title="Conversion Amount ($)",
            yaxis_tickformat="$,.0f",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # MONTE CARLO
    # ==========================================================================
    if run_monte_carlo_sim:
        st.header("🎲 Monte Carlo Results")
        
        mc_results = {
            'S1: Tax-Optimized': mc_s1,
            'S2: After-Tax First': mc_s2,
            'S2 + Roth Conv': mc_s2_roth
        }
        
        mc_summary = []
        for name, mc in mc_results.items():
            if show_real:
                mc_summary.append({
                    'Strategy': name,
                    'Median Final': mc['median_final_real'],
                    '10th Percentile': mc['p10_final_real'],
                    '90th Percentile': mc['p90_final_real'],
                    'Median Taxes': mc['median_taxes_real']
                })
            else:
                mc_summary.append({
                    'Strategy': name,
                    'Median Final': mc['median_final'],
                    '10th Percentile': mc['p10_final'],
                    '90th Percentile': mc['p90_final'],
                    'Median Taxes': mc['median_taxes']
                })
        
        mc_df = pd.DataFrame(mc_summary).sort_values('Median Final', ascending=False)
        
        mc_display = mc_df.copy()
        for col in ['Median Final', '10th Percentile', '90th Percentile', 'Median Taxes']:
            mc_display[col] = mc_display[col].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(mc_display, use_container_width=True, hide_index=True)
    
    # ==========================================================================
    # DETAILED DATA
    # ==========================================================================
    st.header("📊 Detailed Data")
    
    with st.expander("View Year-by-Year Data"):
        detail_strategy = st.selectbox("Select Strategy", list(strategies.keys()), key="detail")
        show_real_detail = st.checkbox("Show Real (Today's $) columns", value=True)
        
        results = strategies[detail_strategy]
        detail_df = pd.DataFrame(results)
        detail_df = detail_df.drop(columns=['new_balances', 'inflation_multiplier', 'year_offset'])
        
        if not show_real_detail:
            real_cols = [c for c in detail_df.columns if c.startswith('real_')]
            detail_df = detail_df.drop(columns=real_cols)
        
        st.dataframe(detail_df, use_container_width=True, height=400)
    
    # ==========================================================================
    # EXPORT
    # ==========================================================================
    st.header("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        export_data = []
        for name, results in strategies.items():
            for r in results:
                row = r.copy()
                row['strategy'] = name
                del row['new_balances']
                export_data.append(row)
        
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="retirement_analysis.csv",
            mime="text/csv"
        )
    
    with col2:
        params_dict = {k: v for k, v in params.__dict__.items()}
        params_df = pd.DataFrame([params_dict])
        params_csv = params_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Parameters CSV",
            data=params_csv,
            file_name="retirement_parameters.csv",
            mime="text/csv"
        )
    
    with col3:
        heir_export = pd.DataFrame(heir_analysis)
        heir_csv = heir_export.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Heir Analysis CSV",
            data=heir_csv,
            file_name="heir_tax_analysis.csv",
            mime="text/csv"
        )
    
    # ==========================================================================
    # FOOTER
    # ==========================================================================
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Retirement Withdrawal Strategy Planner v4 | For educational purposes only | Not financial advice
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
