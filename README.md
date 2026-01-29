# models-vs-code-failure-modes
Compare failure behaviors of deterministic code vs ML on the same task


This project compares how **traditional deterministic code** and a **machine learning model** fail on the *same* prediction task.

Instead of only measuring accuracy, the goal is to analyze **failure behavior**:
- Does it fail loudly (crash/error) or softly (confident but wrong)?
- Are failures predictable or surprising?
- What kinds of edge cases break each approach?

## Plan
- [ ] Implement a deterministic baseline
- [ ] Implement a simple ML baseline
- [ ] Create an evaluation set with edge cases
- [ ] Compare failures and visualize results

## Example Use Cases
This framework could apply to:

- [ ] Email validation (regex vs learned patterns)
- [ ] Spam detection (rule-based vs ML classifier)
- [ ] Input sanitization (whitelist vs anomaly detection)
- [ ] Price prediction (formula-based vs regression model)

## Technologies

- [ ] Python for both implementations
- [ ] Scikit-learn or PyTorch for ML baseline
- [ ] Matplotlib/Seaborn for visualization
- [ ] Pandas for data analysis

 ## Current Status
 In Progress; Setting up project structure and defining the prediction task.
