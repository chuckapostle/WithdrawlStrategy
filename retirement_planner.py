#!/usr/bin/env python3
"""
Retirement Withdrawal Strategy Planner v5
Interactive Streamlit App for comparing withdrawal strategies with:
- Two configurable annuity streams (with start/end dates)
- Two Non-Qualified Plan distributions (balance-based with performance)
- Separate 401K and Roth performance targets
- Flexible after-tax account (fixed income OR market performance)
- Real dollar (inflation-adjusted) analysis
- Heir tax efficiency analysis

Run with: streamlit run retirement_planner.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    
    # NQ Plan 1
    nq1_balance: float
    nq1_start_age: int
    nq1_end_age: int
    nq1_return: float
    nq1_name: str
    
    # NQ Plan 2
    nq2_balance: float
    nq2_start_age: int
    nq2_end_age: int
    nq2_return: float
    nq2_name: str
    
    # Performance assumptions - SEPARATE for each account type
    inflation_rate: float
    real_return_401k: float      # 401K real return
    real_return_roth: float      # Roth real return (can differ)
    return_std_dev: float        # Standard deviation for market returns
    
    # After-tax account options
    after_tax_use_fixed: bool    # True = fixed income, False = market
    after_tax_fixed_return: float  # Fixed income nominal return
    after_tax_market_return: float  # Market real return (if not fixed)
    
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
# NQ PLAN CLASS
# =============================================================================
class NQPlan:
    """Non-Qualified Plan with balance-based distributions"""
    
    def __init__(self, initial_balance: float, start_age: int, end_age: int, 
                 annual_return: float, name: str):
        self.initial_balance = initial_balance
        self.start_age = start_age
        self.end_age = end_age
        self.annual_return = annual_return
        self.name = name
        self.balance = initial_balance
        self.distribution_years = max(1, end_age - start_age + 1)
    
    def reset(self):
        """Reset balance to initial"""
        self.balance = self.initial_balance
    
    def get_distribution(self, age: int, year_return: float = None) -> Tuple[float, float]:
        """
        Get distribution for the year and update balance.
        Returns (distribution_amount, ending_balance)
        
        Distribution = (Balance + Performance) / Remaining Years
        """
        if age < self.start_age or age > self.end_age or self.balance <= 0:
            return 0, self.balance
        
        # Use provided return or default
        actual_return = year_return if year_return is not None else self.annual_return
        
        # Calculate performance on current balance
        performance = self.balance * actual_return
        balance_with_performance = self.balance + performance
        
        # Calculate remaining years including this one
        remaining_years = max(1, self.end_age - age + 1)
        
        # Distribution is balance+performance divided by remaining years
        distribution = balance_with_performance / remaining_years
        
        # Update balance (subtract distribution, performance already added)
        self.balance = balance_with_performance - distribution
        
        return distribution, self.balance

# =============================================================================
# TAX FUNCTIONS
# =============================================================================
def calculate_ss_taxable(ss_income: float, other_income: float) -> float:
    if ss_income == 0:
        return 0
    provisional = other_income + (ss_income * 0.5)
    if provisional > 44_000:
        return ss_income * 0.85
    elif provisional > 32_000:
        return ss_income * 0.50
    return 0

def calculate_federal_tax(taxable_income: float, bracket_12_top: float) -> float:
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
    if age < RMD_START_AGE or balance <= 0:
        return 0
    divisor = RMD_DIVISORS.get(age, 6.0)
    return balance / divisor

def calculate_heir_after_tax_value(
    bal_401k: float, bal_roth: float, bal_after_tax: float,
    heir_federal_rate: float, heir_state_rate: float, cost_basis_pct: float
) -> Dict:
    heir_combined_rate = heir_federal_rate + heir_state_rate
    
    tax_on_401k = bal_401k * heir_combined_rate
    heir_value_401k = bal_401k - tax_on_401k
    
    tax_on_roth = 0
    heir_value_roth = bal_roth
    
    potential_gain = bal_after_tax * 0.10
    tax_on_after_tax = potential_gain * 0.15
    heir_value_after_tax = bal_after_tax - tax_on_after_tax
    
    total_pretax = bal_401k + bal_roth + bal_after_tax
    total_after_tax = heir_value_401k + heir_value_roth + heir_value_after_tax
    total_tax = tax_on_401k + tax_on_roth + tax_on_after_tax
    
    return {
        'bal_401k': bal_401k, 'bal_roth': bal_roth, 'bal_after_tax': bal_after_tax,
        'tax_on_401k': tax_on_401k, 'tax_on_roth': tax_on_roth, 'tax_on_after_tax': tax_on_after_tax,
        'heir_value_401k': heir_value_401k, 'heir_value_roth': heir_value_roth,
        'heir_value_after_tax': heir_value_after_tax, 'total_pretax': total_pretax,
        'total_after_tax_to_heirs': total_after_tax, 'total_heir_tax': total_tax,
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
    nq1: NQPlan,
    nq2: NQPlan,
    returns: Dict[str, float] = None
) -> Dict:
    """
    Simulate one year of retirement.
    
    returns dict can contain:
    - 'return_401k': nominal return for 401K
    - 'return_roth': nominal return for Roth
    - 'return_after_tax': nominal return for after-tax (if market-based)
    - 'return_nq1': nominal return for NQ Plan 1
    - 'return_nq2': nominal return for NQ Plan 2
    """
    bal_401k, bal_roth, bal_after_tax = balances
    
    # Default returns if not provided
    if returns is None:
        returns = {}
    
    nominal_return_401k = returns.get('return_401k', params.real_return_401k + params.inflation_rate)
    nominal_return_roth = returns.get('return_roth', params.real_return_roth + params.inflation_rate)
    
    if params.after_tax_use_fixed:
        nominal_return_after_tax = params.after_tax_fixed_return
    else:
        nominal_return_after_tax = returns.get('return_after_tax', 
                                                params.after_tax_market_return + params.inflation_rate)
    
    nq1_return = returns.get('return_nq1', params.nq1_return)
    nq2_return = returns.get('return_nq2', params.nq2_return)
    
    year_offset = age - params.start_age
    inflation_mult = (1 + params.inflation_rate) ** year_offset
    
    expenses = params.initial_expenses * inflation_mult
    bracket_12 = params.tax_bracket_12_top * inflation_mult
    std_ded = params.standard_deduction * inflation_mult
    
    # Regular income sources
    ss_income = params.social_security if age >= params.ss_start_age else 0
    pension = params.pension if age >= params.pension_start_age else 0
    rental = (params.rental_income * inflation_mult) if age >= params.rental_start_age else 0
    
    # Annuities (fixed nominal)
    annuity1 = params.annuity1_amount if (age >= params.annuity1_start_age and age <= params.annuity1_end_age) else 0
    annuity2 = params.annuity2_amount if (age >= params.annuity2_start_age and age <= params.annuity2_end_age) else 0
    
    # NQ Plan distributions
    nq1_dist, nq1_bal = nq1.get_distribution(age, nq1_return)
    nq2_dist, nq2_bal = nq2.get_distribution(age, nq2_return)
    
    total_other_income = ss_income + pension + rental + annuity1 + annuity2 + nq1_dist + nq2_dist
    rmd = calculate_rmd(age, bal_401k)
    gap = max(0, expenses - total_other_income)
    
    # Tax basis calculation (NQ distributions are ordinary income)
    ss_taxable = calculate_ss_taxable(ss_income, pension + rental + annuity1 + annuity2 + nq1_dist + nq2_dist)
    base_ordinary = ss_taxable + pension + rental + annuity1 + annuity2 + nq1_dist + nq2_dist
    agi_threshold_12 = std_ded + bracket_12
    
    # Strategy-specific withdrawal logic
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
    
    # Calculate taxes
    total_401k_outflow = draw_401k + roth_conversion
    total_ordinary = base_ordinary + total_401k_outflow
    taxable_income = max(0, total_ordinary - std_ded)
    federal_tax = calculate_federal_tax(taxable_income, bracket_12)
    
    cap_gains = draw_after_tax * 0.5 if draw_after_tax > 0 else 0
    cap_gains_tax = cap_gains * 0.15
    total_tax = federal_tax + cap_gains_tax
    
    conversion_tax = roth_conversion * 0.12
    
    # Apply growth with SEPARATE returns
    new_401k = (bal_401k - total_401k_outflow) * (1 + nominal_return_401k)
    new_roth = (bal_roth - draw_roth + roth_conversion) * (1 + nominal_return_roth)
    new_after_tax = (bal_after_tax - draw_after_tax) * (1 + nominal_return_after_tax)
    
    total_balance = new_401k + new_roth + new_after_tax
    
    # Real dollar values
    real_mult = 1 / inflation_mult
    
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
        'nq1_dist': nq1_dist,
        'nq2_dist': nq2_dist,
        'nq1_balance': nq1_bal,
        'nq2_balance': nq2_bal,
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
        # Returns used
        'return_401k': nominal_return_401k,
        'return_roth': nominal_return_roth,
        'return_after_tax': nominal_return_after_tax,
        # Real values
        'real_expenses': expenses * real_mult,
        'real_other_income': total_other_income * real_mult,
        'real_nq1_dist': nq1_dist * real_mult,
        'real_nq2_dist': nq2_dist * real_mult,
        'real_total_tax': total_tax * real_mult,
        'real_bal_401k': new_401k * real_mult,
        'real_bal_roth': new_roth * real_mult,
        'real_bal_after_tax': new_after_tax * real_mult,
        'real_total_balance': total_balance * real_mult,
        'inflation_multiplier': inflation_mult,
        'new_balances': (new_401k, new_roth, new_after_tax)
    }

def run_simulation(params: SimulationParams, strategy: str) -> List[Dict]:
    """Run full deterministic simulation"""
    balances = (params.initial_401k, params.initial_roth, params.initial_after_tax)
    
    # Initialize NQ Plans
    nq1 = NQPlan(params.nq1_balance, params.nq1_start_age, params.nq1_end_age,
                 params.nq1_return, params.nq1_name)
    nq2 = NQPlan(params.nq2_balance, params.nq2_start_age, params.nq2_end_age,
                 params.nq2_return, params.nq2_name)
    
    results = []
    cum_taxes = 0
    cum_taxes_real = 0
    cum_conversions = 0
    cum_conversion_tax = 0
    
    for age in range(params.start_age, params.end_age + 1):
        year_result = simulate_year(age, balances, params, strategy, nq1, nq2)
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
    all_final_401k = []
    all_final_roth = []
    all_final_after_tax = []
    
    total_years = params.end_age - params.start_age + 1
    final_inflation_mult = (1 + params.inflation_rate) ** total_years
    
    for sim in range(params.num_simulations):
        balances = (params.initial_401k, params.initial_roth, params.initial_after_tax)
        
        # Reset NQ Plans for each simulation
        nq1 = NQPlan(params.nq1_balance, params.nq1_start_age, params.nq1_end_age,
                     params.nq1_return, params.nq1_name)
        nq2 = NQPlan(params.nq2_balance, params.nq2_start_age, params.nq2_end_age,
                     params.nq2_return, params.nq2_name)
        
        trajectory = [sum(balances)]
        cum_taxes = 0
        cum_taxes_real = 0
        
        for age in range(params.start_age, params.end_age + 1):
            # Generate random returns with SEPARATE means for each account
            real_return_401k = np.random.normal(params.real_return_401k, params.return_std_dev)
            real_return_roth = np.random.normal(params.real_return_roth, params.return_std_dev)
            
            returns = {
                'return_401k': real_return_401k + params.inflation_rate,
                'return_roth': real_return_roth + params.inflation_rate,
            }
            
            # After-tax: only randomize if market-based
            if not params.after_tax_use_fixed:
                real_return_after_tax = np.random.normal(params.after_tax_market_return, params.return_std_dev)
                returns['return_after_tax'] = real_return_after_tax + params.inflation_rate
            
            # NQ Plans: randomize around their target returns
            returns['return_nq1'] = np.random.normal(params.nq1_return, params.return_std_dev * 0.5)
            returns['return_nq2'] = np.random.normal(params.nq2_return, params.return_std_dev * 0.5)
            
            year_result = simulate_year(age, balances, params, strategy, nq1, nq2, returns)
            cum_taxes += year_result['total_tax']
            cum_taxes_real += year_result['real_total_tax']
            balances = year_result['new_balances']
            trajectory.append(sum(balances))
        
        all_final_balances.append(sum(balances))
        all_final_balances_real.append(sum(balances) / final_inflation_mult)
        all_cum_taxes.append(cum_taxes)
        all_cum_taxes_real.append(cum_taxes_real)
        all_trajectories.append(trajectory)
        all_final_401k.append(balances[0])
        all_final_roth.append(balances[1])
        all_final_after_tax.append(balances[2])
    
    return {
        'final_balances': np.array(all_final_balances),
        'final_balances_real': np.array(all_final_balances_real),
        'cum_taxes': np.array(all_cum_taxes),
        'cum_taxes_real': np.array(all_cum_taxes_real),
        'trajectories': np.array(all_trajectories),
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
    st.title("📊 Retirement Withdrawal Strategy Planner v5")
    st.markdown("*With NQ Plan distributions, separate account returns & flexible after-tax options*")
    
    # ==========================================================================
    # SIDEBAR
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
        annuity1_end_age = st.number_input("End Age", value=72, min_value=55, max_value=100, key="ann1_end")
        if annuity1_amount > 0:
            st.caption(f"📅 {annuity1_name}: Ages {annuity1_start_age}-{annuity1_end_age}")
        
        st.divider()
        
        # Annuity 2
        st.subheader("💵 Annuity 2")
        annuity2_name = st.text_input("Name", value="Annuity 2", key="ann2_name")
        col1, col2 = st.columns(2)
        with col1:
            annuity2_amount = st.number_input("Annual Amount ($)", value=0, step=5_000, format="%d", key="ann2_amt")
        with col2:
            annuity2_start_age = st.number_input("Start Age", value=70, min_value=55, max_value=95, key="ann2_start")
        annuity2_end_age = st.number_input("End Age", value=79, min_value=55, max_value=100, key="ann2_end")
        if annuity2_amount > 0:
            st.caption(f"📅 {annuity2_name}: Ages {annuity2_start_age}-{annuity2_end_age}")
        
        st.divider()
        
        # NQ Plan 1
        st.subheader("📋 Non-Qualified Plan 1")
        nq1_name = st.text_input("Name", value="NQ Plan 1", key="nq1_name")
        nq1_balance = st.number_input("Starting Balance ($)", value=0, step=50_000, format="%d", key="nq1_bal")
        col1, col2 = st.columns(2)
        with col1:
            nq1_start_age = st.number_input("Start Age", value=63, min_value=55, max_value=95, key="nq1_start")
        with col2:
            nq1_end_age = st.number_input("End Age", value=72, min_value=55, max_value=100, key="nq1_end")
        nq1_return = st.slider("Annual Return (%)", 0.0, 12.0, 5.0, 0.5, key="nq1_ret") / 100
        if nq1_balance > 0:
            nq1_years = nq1_end_age - nq1_start_age + 1
            st.caption(f"📅 {nq1_name}: ${nq1_balance:,.0f} over {nq1_years} years @ {nq1_return*100:.1f}%")
        
        st.divider()
        
        # NQ Plan 2
        st.subheader("📋 Non-Qualified Plan 2")
        nq2_name = st.text_input("Name", value="NQ Plan 2", key="nq2_name")
        nq2_balance = st.number_input("Starting Balance ($)", value=0, step=50_000, format="%d", key="nq2_bal")
        col1, col2 = st.columns(2)
        with col1:
            nq2_start_age = st.number_input("Start Age", value=70, min_value=55, max_value=95, key="nq2_start")
        with col2:
            nq2_end_age = st.number_input("End Age", value=79, min_value=55, max_value=100, key="nq2_end")
        nq2_return = st.slider("Annual Return (%)", 0.0, 12.0, 5.0, 0.5, key="nq2_ret") / 100
        if nq2_balance > 0:
            nq2_years = nq2_end_age - nq2_start_age + 1
            st.caption(f"📅 {nq2_name}: ${nq2_balance:,.0f} over {nq2_years} years @ {nq2_return*100:.1f}%")
        
        st.divider()
        
        # Expenses
        st.subheader("💸 Expenses")
        initial_expenses = st.number_input("Annual Expenses ($)", value=150_000, step=5_000, format="%d")
        
        st.divider()
        
        # Performance Assumptions - SEPARATE
        st.subheader("📊 Performance Assumptions")
        inflation_rate = st.slider("Inflation Rate (%)", 0.0, 5.0, 2.66, 0.1) / 100
        
        st.markdown("**Tax-Advantaged Accounts**")
        real_return_401k = st.slider("401K Real Return (%)", 0.0, 12.0, 6.0, 0.5, 
                                     help="Real (after-inflation) return for 401K") / 100
        real_return_roth = st.slider("Roth Real Return (%)", 0.0, 12.0, 6.0, 0.5,
                                     help="Real (after-inflation) return for Roth") / 100
        
        return_std_dev = st.slider("Return Std Deviation (%)", 5.0, 25.0, 12.0, 1.0,
                                   help="Volatility for Monte Carlo simulations") / 100
        
        st.markdown("**After-Tax Account**")
        after_tax_use_fixed = st.radio(
            "After-Tax Investment Type",
            options=[True, False],
            format_func=lambda x: "Fixed Income (stable)" if x else "Market-Based (variable)",
            help="Choose fixed income for stable returns, or market-based for variable returns"
        )
        
        if after_tax_use_fixed:
            after_tax_fixed_return = st.slider("Fixed Income Return (%)", 0.0, 8.0, 5.0, 0.25) / 100
            after_tax_market_return = 0.06  # Default, not used
        else:
            after_tax_fixed_return = 0.05  # Default, not used
            after_tax_market_return = st.slider("After-Tax Market Return (%)", 0.0, 12.0, 6.0, 0.5,
                                                help="Subject to same volatility as other accounts") / 100
        
        st.divider()
        
        # Tax Parameters
        st.subheader("🏛️ Tax Parameters")
        tax_bracket_12_top = st.number_input("12% Bracket Top ($)", value=97_000, step=1_000, format="%d")
        standard_deduction = st.number_input("Standard Deduction ($)", value=32_500, step=500, format="%d")
        
        st.divider()
        
        # Heir Tax
        st.subheader("👨‍👩‍👧‍👦 Heir Tax Assumptions")
        heir_federal_rate = st.slider("Heir Federal Rate (%)", 10, 37, 24, 1) / 100
        heir_state_rate = st.slider("Heir State Rate (%)", 0, 13, 5, 1) / 100
        after_tax_cost_basis_pct = st.slider("After-Tax Cost Basis (%)", 30, 90, 50, 5) / 100
        
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
        run_monte_carlo_sim = st.checkbox("Run Monte Carlo", value=False)
        if run_monte_carlo_sim:
            num_simulations = st.slider("Simulations", 100, 5000, 1000, 100)
        else:
            num_simulations = 1000
    
    # Build params
    params = SimulationParams(
        initial_401k=initial_401k, initial_roth=initial_roth, initial_after_tax=initial_after_tax,
        social_security=social_security, ss_start_age=ss_start_age,
        pension=pension, pension_start_age=pension_start_age,
        rental_income=rental_income, rental_start_age=rental_start_age,
        annuity1_amount=annuity1_amount, annuity1_start_age=annuity1_start_age,
        annuity1_end_age=annuity1_end_age, annuity1_name=annuity1_name,
        annuity2_amount=annuity2_amount, annuity2_start_age=annuity2_start_age,
        annuity2_end_age=annuity2_end_age, annuity2_name=annuity2_name,
        nq1_balance=nq1_balance, nq1_start_age=nq1_start_age, nq1_end_age=nq1_end_age,
        nq1_return=nq1_return, nq1_name=nq1_name,
        nq2_balance=nq2_balance, nq2_start_age=nq2_start_age, nq2_end_age=nq2_end_age,
        nq2_return=nq2_return, nq2_name=nq2_name,
        inflation_rate=inflation_rate, real_return_401k=real_return_401k,
        real_return_roth=real_return_roth, return_std_dev=return_std_dev,
        after_tax_use_fixed=after_tax_use_fixed, after_tax_fixed_return=after_tax_fixed_return,
        after_tax_market_return=after_tax_market_return, initial_expenses=initial_expenses,
        tax_bracket_12_top=tax_bracket_12_top, standard_deduction=standard_deduction,
        heir_federal_rate=heir_federal_rate, heir_state_rate=heir_state_rate,
        after_tax_cost_basis_pct=after_tax_cost_basis_pct,
        start_age=start_age, end_age=end_age,
        run_monte_carlo=run_monte_carlo_sim, num_simulations=num_simulations
    )
    
    # Update params with expenses is no longer needed - expenses is now defined before params
    
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
    
    total_years = end_age - start_age
    final_inflation_mult = (1 + inflation_rate) ** total_years
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    st.header("📋 Summary Results")
    
    show_real = st.toggle("Show in Today's Dollars (Real)", value=False)
    dollar_label = "Today's $" if show_real else "Nominal $"
    
    # Show return assumptions
    with st.expander("📊 Performance Assumptions Used"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("401K Real Return", f"{real_return_401k*100:.1f}%")
            st.metric("401K Nominal Return", f"{(real_return_401k + inflation_rate)*100:.1f}%")
        with col2:
            st.metric("Roth Real Return", f"{real_return_roth*100:.1f}%")
            st.metric("Roth Nominal Return", f"{(real_return_roth + inflation_rate)*100:.1f}%")
        with col3:
            if after_tax_use_fixed:
                st.metric("After-Tax Type", "Fixed Income")
                st.metric("After-Tax Return", f"{after_tax_fixed_return*100:.1f}%")
            else:
                st.metric("After-Tax Type", "Market-Based")
                st.metric("After-Tax Real Return", f"{after_tax_market_return*100:.1f}%")
    
    # Summary table
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
    
    summary_df = pd.DataFrame(summary_data).sort_values('Final Balance', ascending=False)
    
    col1, col2, col3, col4 = st.columns(4)
    winner = summary_df.iloc[0]
    with col1:
        st.metric("🏆 Best Strategy", winner['Strategy'].split(':')[0] if ':' in winner['Strategy'] else winner['Strategy'])
    with col2:
        st.metric(f"Final Balance ({dollar_label})", f"${winner['Final Balance']:,.0f}")
    with col3:
        st.metric(f"Total Taxes ({dollar_label})", f"${winner['Total Taxes']:,.0f}")
    with col4:
        st.metric("Roth Converted", f"${winner['Total Conversions']:,.0f}")
    
    display_df = summary_df.copy()
    for col in ['Final Balance', 'Final 401K', 'Final Roth', 'Final After-Tax', 'Total Taxes', 'Total Conversions']:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # ==========================================================================
    # NQ PLAN DISTRIBUTIONS
    # ==========================================================================
    if nq1_balance > 0 or nq2_balance > 0:
        st.header("📋 Non-Qualified Plan Distributions")
        
        nq_data = []
        for r in results_s1:
            if r['nq1_dist'] > 0 or r['nq2_dist'] > 0:
                if show_real:
                    nq_data.append({
                        'Age': r['age'],
                        f'{nq1_name} Dist': r['real_nq1_dist'],
                        f'{nq1_name} Bal': r['nq1_balance'] / r['inflation_multiplier'],
                        f'{nq2_name} Dist': r['real_nq2_dist'],
                        f'{nq2_name} Bal': r['nq2_balance'] / r['inflation_multiplier'],
                    })
                else:
                    nq_data.append({
                        'Age': r['age'],
                        f'{nq1_name} Dist': r['nq1_dist'],
                        f'{nq1_name} Bal': r['nq1_balance'],
                        f'{nq2_name} Dist': r['nq2_dist'],
                        f'{nq2_name} Bal': r['nq2_balance'],
                    })
        
        if nq_data:
            nq_df = pd.DataFrame(nq_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                if nq1_balance > 0:
                    fig.add_trace(go.Bar(x=nq_df['Age'], y=nq_df[f'{nq1_name} Dist'],
                                        name=nq1_name, marker_color='#7B1FA2'))
                if nq2_balance > 0:
                    fig.add_trace(go.Bar(x=nq_df['Age'], y=nq_df[f'{nq2_name} Dist'],
                                        name=nq2_name, marker_color='#00897B'))
                fig.update_layout(
                    title=f"NQ Plan Distributions ({dollar_label})",
                    barmode='group', yaxis_tickformat="$,.0f", height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                if nq1_balance > 0:
                    fig.add_trace(go.Scatter(x=nq_df['Age'], y=nq_df[f'{nq1_name} Bal'],
                                            name=nq1_name, line=dict(color='#7B1FA2', width=2)))
                if nq2_balance > 0:
                    fig.add_trace(go.Scatter(x=nq_df['Age'], y=nq_df[f'{nq2_name} Bal'],
                                            name=nq2_name, line=dict(color='#00897B', width=2)))
                fig.update_layout(
                    title=f"NQ Plan Remaining Balances ({dollar_label})",
                    yaxis_tickformat="$,.0f", height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            if nq1_balance > 0:
                total_nq1_dist = sum(r['nq1_dist'] for r in results_s1)
                st.metric(f"Total {nq1_name} Distributions", 
                         f"${total_nq1_dist:,.0f}" if not show_real else f"${total_nq1_dist/final_inflation_mult:,.0f}")
            if nq2_balance > 0:
                total_nq2_dist = sum(r['nq2_dist'] for r in results_s1)
                st.metric(f"Total {nq2_name} Distributions",
                         f"${total_nq2_dist:,.0f}" if not show_real else f"${total_nq2_dist/final_inflation_mult:,.0f}")
    
    # ==========================================================================
    # REAL DOLLAR ANALYSIS
    # ==========================================================================
    st.header("💵 Real Dollar Analysis")
    
    st.markdown(f"""
    *At {inflation_rate*100:.2f}% inflation over {total_years} years: 
    $1 today = ${final_inflation_mult:.2f} nominal at age {end_age}*
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        comp_data = []
        for name, results in strategies.items():
            final = results[-1]
            comp_data.append({
                'Strategy': name,
                'Nominal': final['total_balance'],
                "Real": final['real_total_balance'],
            })
        comp_df = pd.DataFrame(comp_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Nominal $', x=comp_df['Strategy'], y=comp_df['Nominal'],
                            marker_color='#1976D2', text=[f"${x/1e6:.1f}M" for x in comp_df['Nominal']]))
        fig.add_trace(go.Bar(name="Real (Today's $)", x=comp_df['Strategy'], y=comp_df['Real'],
                            marker_color='#388E3C', text=[f"${x/1e6:.1f}M" for x in comp_df['Real']]))
        fig.update_layout(barmode='group', yaxis_tickformat="$,.0f", height=400,
                         title="Nominal vs Real Final Balances")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        ages = [r['age'] for r in results_s1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ages, y=[r['total_balance'] for r in results_s1],
                                name='Nominal', line=dict(color='#1976D2', dash='dash')))
        fig.add_trace(go.Scatter(x=ages, y=[r['real_total_balance'] for r in results_s1],
                                name="Real (Today's $)", line=dict(color='#388E3C', width=3)))
        fig.update_layout(yaxis_tickformat="$,.0f", height=400,
                         title="S1: Nominal vs Real Value Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # HEIR ANALYSIS
    # ==========================================================================
    st.header("👨‍👩‍👧‍👦 Heir Tax Analysis")
    
    heir_analysis = []
    for name, results in strategies.items():
        final = results[-1]
        heir_calc = calculate_heir_after_tax_value(
            final['bal_401k'], final['bal_roth'], final['bal_after_tax'],
            heir_federal_rate, heir_state_rate, after_tax_cost_basis_pct
        )
        heir_calc['strategy'] = name
        heir_calc['real_to_heirs'] = heir_calc['total_after_tax_to_heirs'] / final_inflation_mult
        heir_analysis.append(heir_calc)
    
    heir_df = pd.DataFrame(heir_analysis)
    
    col1, col2 = st.columns(2)
    
    with col1:
        heir_compare = []
        for _, row in heir_df.iterrows():
            val = row['real_to_heirs'] if show_real else row['total_after_tax_to_heirs']
            heir_compare.append({
                'Strategy': row['strategy'],
                'Pre-Tax Legacy': row['total_pretax'] / (final_inflation_mult if show_real else 1),
                'Heir Tax': row['total_heir_tax'] / (final_inflation_mult if show_real else 1),
                'After-Tax to Heirs': val,
            })
        heir_compare_df = pd.DataFrame(heir_compare).sort_values('After-Tax to Heirs', ascending=False)
        
        display_heir = heir_compare_df.copy()
        for col in ['Pre-Tax Legacy', 'Heir Tax', 'After-Tax to Heirs']:
            display_heir[col] = display_heir[col].apply(lambda x: f"${x:,.0f}")
        st.dataframe(display_heir, use_container_width=True, hide_index=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='401K (taxable)', x=heir_df['strategy'],
                            y=heir_df['bal_401k']/(final_inflation_mult if show_real else 1),
                            marker_color='#EF5350'))
        fig.add_trace(go.Bar(name='Roth (tax-free)', x=heir_df['strategy'],
                            y=heir_df['bal_roth']/(final_inflation_mult if show_real else 1),
                            marker_color='#66BB6A'))
        fig.add_trace(go.Bar(name='After-Tax', x=heir_df['strategy'],
                            y=heir_df['bal_after_tax']/(final_inflation_mult if show_real else 1),
                            marker_color='#FFA726'))
        fig.update_layout(barmode='stack', yaxis_tickformat="$,.0f", height=400,
                         title=f"Legacy by Account Type ({dollar_label})")
        st.plotly_chart(fig, use_container_width=True)
    
    # Roth Conversion ROI
    st.subheader("🔄 Roth Conversion ROI")
    
    s2_final = results_s2[-1]
    s2r_final = results_s2_roth[-1]
    
    s2_heir = calculate_heir_after_tax_value(s2_final['bal_401k'], s2_final['bal_roth'], 
                                             s2_final['bal_after_tax'], heir_federal_rate, 
                                             heir_state_rate, after_tax_cost_basis_pct)
    s2r_heir = calculate_heir_after_tax_value(s2r_final['bal_401k'], s2r_final['bal_roth'],
                                              s2r_final['bal_after_tax'], heir_federal_rate,
                                              heir_state_rate, after_tax_cost_basis_pct)
    
    extra_taxes = s2r_final['cum_taxes'] - s2_final['cum_taxes']
    extra_to_heirs = s2r_heir['total_after_tax_to_heirs'] - s2_heir['total_after_tax_to_heirs']
    net_benefit = extra_to_heirs - extra_taxes
    roi = (extra_to_heirs / extra_taxes - 1) * 100 if extra_taxes > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Extra Tax Paid", f"${extra_taxes:,.0f}")
    with col2:
        st.metric("Extra to Heirs", f"${extra_to_heirs:,.0f}")
    with col3:
        st.metric("Net Benefit", f"${net_benefit:,.0f}",
                 delta="Worth it ✅" if net_benefit > 0 else "Not worth it ❌")
    
    # ==========================================================================
    # CHARTS
    # ==========================================================================
    st.header("📈 Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Balances", "Income", "Taxes", "Conversions"])
    
    colors = {'S1: Tax-Optimized': '#2E7D32', 'S2: After-Tax First': '#E65100', 'S2 + Roth Conv': '#7B1FA2'}
    
    with tab1:
        fig = go.Figure()
        for name, results in strategies.items():
            ages = [r['age'] for r in results]
            vals = [r['real_total_balance'] if show_real else r['total_balance'] for r in results]
            fig.add_trace(go.Scatter(x=ages, y=vals, name=name, line=dict(color=colors[name], width=2)))
        fig.add_vline(x=75, line_dash="dot", line_color="red", annotation_text="RMD Start")
        fig.update_layout(title=f"Total Balance ({dollar_label})", yaxis_tickformat="$,.0f", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        income_data = []
        for r in results_s1:
            mult = r['inflation_multiplier'] if not show_real else 1
            income_data.append({
                'Age': r['age'],
                'SS': r['ss_income'] / mult,
                'Pension': r['pension'] / mult,
                'Rental': r['rental'] / mult,
                annuity1_name: r['annuity1'] / mult,
                annuity2_name: r['annuity2'] / mult,
                nq1_name: r['nq1_dist'] / mult,
                nq2_name: r['nq2_dist'] / mult,
            })
        income_df = pd.DataFrame(income_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=income_df['Age'], y=income_df['SS'], name='SS', marker_color='#1976D2'))
        fig.add_trace(go.Bar(x=income_df['Age'], y=income_df['Pension'], name='Pension', marker_color='#388E3C'))
        fig.add_trace(go.Bar(x=income_df['Age'], y=income_df['Rental'], name='Rental', marker_color='#F57C00'))
        if annuity1_amount > 0:
            fig.add_trace(go.Bar(x=income_df['Age'], y=income_df[annuity1_name], name=annuity1_name, marker_color='#7B1FA2'))
        if annuity2_amount > 0:
            fig.add_trace(go.Bar(x=income_df['Age'], y=income_df[annuity2_name], name=annuity2_name, marker_color='#00897B'))
        if nq1_balance > 0:
            fig.add_trace(go.Bar(x=income_df['Age'], y=income_df[nq1_name], name=nq1_name, marker_color='#E91E63'))
        if nq2_balance > 0:
            fig.add_trace(go.Bar(x=income_df['Age'], y=income_df[nq2_name], name=nq2_name, marker_color='#9C27B0'))
        
        fig.update_layout(barmode='stack', title=f"Income Sources ({dollar_label})", 
                         yaxis_tickformat="$,.0f", height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = go.Figure()
        for name, results in strategies.items():
            ages = [r['age'] for r in results]
            vals = [r['cum_taxes_real'] if show_real else r['cum_taxes'] for r in results]
            fig.add_trace(go.Scatter(x=ages, y=vals, name=name, line=dict(color=colors[name], width=2)))
        fig.update_layout(title=f"Cumulative Taxes ({dollar_label})", yaxis_tickformat="$,.0f", height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        fig = go.Figure()
        ages = [r['age'] for r in results_s2_roth]
        vals = [r['roth_conversion'] / (r['inflation_multiplier'] if show_real else 1) for r in results_s2_roth]
        fig.add_trace(go.Bar(x=ages, y=vals, name='Conversions', marker_color='#7B1FA2'))
        fig.update_layout(title=f"Annual Roth Conversions ({dollar_label})", yaxis_tickformat="$,.0f", height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # MONTE CARLO
    # ==========================================================================
    if run_monte_carlo_sim:
        st.header("🎲 Monte Carlo Results")
        
        mc_results = {'S1: Tax-Optimized': mc_s1, 'S2: After-Tax First': mc_s2, 'S2 + Roth Conv': mc_s2_roth}
        
        mc_summary = []
        for name, mc in mc_results.items():
            if show_real:
                mc_summary.append({
                    'Strategy': name,
                    'Median': mc['median_final_real'],
                    '10th Pct': mc['p10_final_real'],
                    '90th Pct': mc['p90_final_real'],
                })
            else:
                mc_summary.append({
                    'Strategy': name,
                    'Median': mc['median_final'],
                    '10th Pct': mc['p10_final'],
                    '90th Pct': mc['p90_final'],
                })
        
        mc_df = pd.DataFrame(mc_summary).sort_values('Median', ascending=False)
        mc_display = mc_df.copy()
        for col in ['Median', '10th Pct', '90th Pct']:
            mc_display[col] = mc_display[col].apply(lambda x: f"${x:,.0f}")
        st.dataframe(mc_display, use_container_width=True, hide_index=True)
        
        fig = go.Figure()
        for name, mc in mc_results.items():
            vals = mc['final_balances_real'] if show_real else mc['final_balances']
            fig.add_trace(go.Histogram(x=vals, name=name, opacity=0.6, nbinsx=50))
        fig.update_layout(barmode='overlay', xaxis_tickformat="$,.0f", height=400,
                         title=f"Distribution of Final Balances ({dollar_label})")
        st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # EXPORT
    # ==========================================================================
    st.header("💾 Export")
    
    col1, col2 = st.columns(2)
    with col1:
        export_data = []
        for name, results in strategies.items():
            for r in results:
                row = {k: v for k, v in r.items() if k != 'new_balances'}
                row['strategy'] = name
                export_data.append(row)
        csv = pd.DataFrame(export_data).to_csv(index=False)
        st.download_button("📥 Download Results CSV", csv, "retirement_analysis.csv", "text/csv")
    
    with col2:
        params_dict = {k: v for k, v in params.__dict__.items()}
        csv = pd.DataFrame([params_dict]).to_csv(index=False)
        st.download_button("📥 Download Parameters CSV", csv, "parameters.csv", "text/csv")
    
    st.divider()
    st.markdown("<div style='text-align:center;color:gray;font-size:0.8em;'>v5 | Educational purposes only | Not financial advice</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
