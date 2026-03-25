import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 設定 Streamlit 頁面
st.set_page_config(page_title="Bandit Strategies Comparison", layout="wide")

st.title("🎰 Comparing 6 Bandit Strategies (Multi-Armed Bandits)")
st.markdown("""
### 🧩 Problem Setup
* **Total budget:** $10,000 (10,000 rounds)
* **Bandits True Means:** Arm A: **0.8**, Arm B: **0.7**, Arm C: **0.5**
* **Goal:** Maximize total expected reward / Minimize cumulative regret.
""")

# 定義環境變數
BUDGET = 10000
TRUE_MEANS = [0.8, 0.7, 0.5]
OPTIMAL_MEAN = max(TRUE_MEANS)

# ==========================================
# 模擬演算法 (Simulation Functions)
# ==========================================
def simulate_ab_test():
    # A/B Testing: First $2,000 allocated equally to A and B (1000 each). Ignore C.
    # Remaining $8,000 to the best empirical mean.
    rewards = np.zeros(BUDGET)
    
    # Exploration Phase
    pulls_A = np.random.binomial(1, TRUE_MEANS[0], 1000)
    pulls_B = np.random.binomial(1, TRUE_MEANS[1], 1000)
    rewards[0:1000] = pulls_A
    rewards[1000:2000] = pulls_B
    
    mean_A = np.mean(pulls_A)
    mean_B = np.mean(pulls_B)
    
    # Exploitation Phase
    best_arm = 0 if mean_A >= mean_B else 1
    rewards[2000:] = np.random.binomial(1, TRUE_MEANS[best_arm], BUDGET - 2000)
    
    return rewards

def simulate_optimistic_initial_values():
    # Initialize Q heavily optimistic (e.g., 5.0)
    Q = np.array([5.0, 5.0, 5.0])
    N = np.zeros(3)
    rewards = np.zeros(BUDGET)
    
    for t in range(BUDGET):
        action = np.argmax(Q)
        reward = np.random.binomial(1, TRUE_MEANS[action])
        rewards[t] = reward
        N[action] += 1
        Q[action] = Q[action] + (1/N[action]) * (reward - Q[action])
    return rewards

def simulate_epsilon_greedy(epsilon=0.1):
    Q = np.zeros(3)
    N = np.zeros(3)
    rewards = np.zeros(BUDGET)
    
    for t in range(BUDGET):
        if np.random.rand() < epsilon:
            action = np.random.choice(3)
        else:
            action = np.argmax(Q)
            
        reward = np.random.binomial(1, TRUE_MEANS[action])
        rewards[t] = reward
        N[action] += 1
        Q[action] = Q[action] + (1/N[action]) * (reward - Q[action])
    return rewards

def simulate_softmax(tau=0.1):
    Q = np.zeros(3)
    N = np.zeros(3)
    rewards = np.zeros(BUDGET)
    
    for t in range(BUDGET):
        # Prevent overflow
        exp_Q = np.exp((Q - np.max(Q)) / tau) 
        probs = exp_Q / np.sum(exp_Q)
        action = np.random.choice(3, p=probs)
        
        reward = np.random.binomial(1, TRUE_MEANS[action])
        rewards[t] = reward
        N[action] += 1
        Q[action] = Q[action] + (1/N[action]) * (reward - Q[action])
    return rewards

def simulate_ucb(c=2.0):
    Q = np.zeros(3)
    N = np.zeros(3)
    rewards = np.zeros(BUDGET)
    
    # Play each arm once
    for t in range(3):
        action = t
        reward = np.random.binomial(1, TRUE_MEANS[action])
        rewards[t] = reward
        N[action] += 1
        Q[action] = reward
        
    for t in range(3, BUDGET):
        ucb_values = Q + c * np.sqrt(np.log(t) / N)
        action = np.argmax(ucb_values)
        
        reward = np.random.binomial(1, TRUE_MEANS[action])
        rewards[t] = reward
        N[action] += 1
        Q[action] = Q[action] + (1/N[action]) * (reward - Q[action])
    return rewards

def simulate_thompson_sampling():
    alpha = np.ones(3)
    beta = np.ones(3)
    rewards = np.zeros(BUDGET)
    
    for t in range(BUDGET):
        sampled_theta = np.random.beta(alpha, beta)
        action = np.argmax(sampled_theta)
        
        reward = np.random.binomial(1, TRUE_MEANS[action])
        rewards[t] = reward
        if reward == 1:
            alpha[action] += 1
        else:
            beta[action] += 1
    return rewards

# ==========================================
# 執行模擬與收集數據
# ==========================================
st.sidebar.header("Simulation Settings")
runs = st.sidebar.slider("Number of Simulations to Average", min_value=1, max_value=50, value=10)

if st.button("Run Simulation"):
    with st.spinner('Simulating 10,000 rounds for 6 algorithms...'):
        results = {
            "A/B Testing": np.zeros(BUDGET),
            "Optimistic Init": np.zeros(BUDGET),
            "ε-Greedy (ε=0.1)": np.zeros(BUDGET),
            "Softmax (τ=0.1)": np.zeros(BUDGET),
            "UCB (c=2)": np.zeros(BUDGET),
            "Thompson Sampling": np.zeros(BUDGET)
        }
        
        for _ in range(runs):
            results["A/B Testing"] += simulate_ab_test()
            results["Optimistic Init"] += simulate_optimistic_initial_values()
            results["ε-Greedy (ε=0.1)"] += simulate_epsilon_greedy()
            results["Softmax (τ=0.1)"] += simulate_softmax()
            results["UCB (c=2)"] += simulate_ucb()
            results["Thompson Sampling"] += simulate_thompson_sampling()
            
        # 平均結果
        for key in results:
            results[key] /= runs

        # 計算遺憾值 (Regret)
        optimal_cumulative_reward = np.cumsum(np.full(BUDGET, OPTIMAL_MEAN))
        
        regret_results = {}
        total_rewards = {}
        for key in results:
            cumulative_reward = np.cumsum(results[key])
            regret_results[key] = optimal_cumulative_reward - cumulative_reward
            total_rewards[key] = cumulative_reward[-1]
            
        # ==========================================
        # 繪製圖表 (類似使用者上傳的圖片)
        # ==========================================
        st.subheader("📊 Performance Visualization (Cumulative Regret)")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, regret in regret_results.items():
            ax.plot(regret, label=key, linewidth=1.5)
            
        ax.set_title(f"Simulated Bandit Performance (Averaged over {runs} runs)")
        ax.set_xlabel("Round Index (Budget)")
        ax.set_ylabel("Cumulative Expected Regret")
        ax.set_yscale('log') # 仿照範例圖片使用對數尺度
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        st.pyplot(fig)

        # ==========================================
        # 建立比較表格 (Step 4)
        # ==========================================
        st.subheader("🔄 Step 4: Class Comparison Table")
        table_data = {
            "Method": [
                "A/B Test", "Optimistic Initial", "ε-Greedy", 
                "Softmax", "UCB", "Thompson Sampling"
            ],
            "Exploration Style": [
                "Static (Fixed budget initially)", "Implicit (Front-loaded)", 
                "Random (Continuous)", "Probabilistic (Temperature-based)", 
                "Confidence-based", "Bayesian (Probability matching)"
            ],
            "Total Reward (Simulated)": [round(total_rewards[k], 2) for k in results.keys()],
            "Regret (Simulated)": [round(regret_results[k][-1], 2) for k in results.keys()],
            "Notes": [
                "Simple but wasteful (Doesn't adapt)", 
                "No exploration parameters needed, but slow to adapt if environment changes", 
                "Easy baseline, but explores sub-optimal arms forever", 
                "Smooth control, but needs tuning of Temperature (τ)", 
                "Efficient, mathematically proven regret bounds", 
                "Best practical, balances exploration/exploitation naturally"
            ]
        }
        df = pd.DataFrame(table_data)
        st.dataframe(df, hide_index=True)

# ==========================================
# 討論與問答 (Step 3 & Step 5)
# ==========================================
st.markdown("---")
st.header("🗣️ Step 5: Discussion Questions & Insights")

tab1, tab2, tab3 = st.tabs(["🏆 Performance & Waste", "💡 Real-World Deployment", "📉 Edge Cases"])

with tab1:
    st.markdown("""
    ### Which method performed best? Why?
    **Thompson Sampling** and **UCB** generally perform the best. 
    * **Thompson Sampling** works well because it uses Bayesian updating to continuously refine its confidence. If an arm proves to be bad, the probability of choosing it drops to near zero very quickly.
    * **UCB** focuses on the *Upper Confidence Bound* ($Q_a + c \\sqrt{\\frac{\\ln t}{N_a}}$), meaning it purely mathematical drives exploration toward arms we are uncertain about, but exploits known winners efficiently.

    ### Which method wastes the most budget?
    **A/B Testing**. In our specific prompt setup, we burned 1,000 pulls ($1,000) on Arm B (True Mean 0.7) during the exploration phase, even if it became obvious very early that Arm A (0.8) was better. Even worse, if Arm C was part of the test, it would have wasted money on a 0.5 return arm.
    
    ### Why is A/B testing not adaptive?
    Because the exploration and exploitation phases are strictly separated. During the first 2,000 rounds, it **does not learn** to stop pulling the worse arm. It rigidly sticks to the schedule, whereas Bandit algorithms adaptively shift traffic to the winning arm *during* the test.
    """)

with tab2:
    st.markdown("""
    ### Which method would you deploy in an Ads system?
    **Thompson Sampling** or **ε-Greedy**. Ads environments are highly stochastic and can change over time. Thompson Sampling is the industry standard for Ads (used heavily by Google and Meta) because it handles uncertainty beautifully and can easily incorporate prior knowledge (Bayesian priors). ε-Greedy is also used for its computational simplicity.

    ### Which method would you deploy in a Clinical trial?
    **Thompson Sampling**. In a clinical trial, "wasting budget" means giving a patient a less effective drug. You want to maximize the number of patients getting the best treatment *as quickly as possible* while still exploring. Thompson Sampling smoothly shifts patients to the winning drug much safer and faster than A/B testing, minimizing patient harm (regret).
    """)

with tab3:
    st.markdown("""
    ### What happens if the budget is smaller?
    If the budget is very small (e.g., 500 rounds), **A/B testing** or **Optimistic Initial Values** might fail completely. A/B testing might not have statistical significance to pick the right arm. **Thompson Sampling** handles small budgets well because it acts optimally given whatever small evidence it has.

    ### What happens if means are closer (e.g., 0.8 vs 0.79)?
    The **Regret** will grow slower for any single mistake (because the penalty is only 0.01 instead of 0.1). However, algorithms like **ε-Greedy** or **A/B testing** will struggle to distinguish the winner. **UCB** will require a very long time to separate the confidence bounds. **Softmax** might just sample them almost equally.
    """)

st.markdown("""
---
*🤖 Instructor Insight Summary:* A/B testing is the simplest but most inefficient. ε-greedy & Softmax serve as good, easy-to-implement baselines. However, **UCB and Thompson Sampling provide the best balance**, with Thompson Sampling often winning in real-world practical applications.
""")