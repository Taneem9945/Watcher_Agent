Not redundant, but it can become redundant or biased if we use it badly.
What these fields are doing now
prediction is the class the model picked, usually benign or attack.
attack_probability is the model’s confidence score for the attack class.
risk_level is just a human-readable bucket derived from attack_probability using fixed thresholds.
In the current code:
prediction comes from torch.argmax(logits, dim=1)
attack_probability comes from softmax(logits)[:, 1]
risk_level comes from thresholding that probability in [src/context_builder.py](d:/Apps/Watcher Agent/src/context_builder.py)
So these numbers are not measured from the network itself. They are model outputs.
How they’re computed
The model produces raw scores called logits.
softmax turns those logits into probabilities that sum to 1.
The attack probability is the probability assigned to class 1.
Then we map that to a label and risk bucket.
So the chain is:
raw flow window -> Mamba model -> logits -> probability -> prediction -> risk level
Does this skew the LLM?
Yes, it can.
If the LLM sees attack_probability: 0.97, it may anchor on that and pay less attention to the actual evidence.
That means the current context packet is good for a first prototype, but it is not ideal if you want the LLM to reason independently.
Why it still helps
The LLM is not supposed to be the detector.
It is supposed to:summarize
interpret
explain
recommend next steps

The model score is one piece of evidence, like an alert severity from a detector.
So the LLM is still useful if the job is:
“Here’s what the detector saw, explain it”
not
“Look at raw data with no guidance and invent a verdict”
The real tradeoff
Including these fields helps the LLM stay grounded.
But it also biases the LLM toward the model’s answer.
That’s fine for an alerting pipeline, but less good if you want pure independent reasoning.
Best practice
For Option 2 or a more serious watcher design, I would usually do one of these:
Give the LLM the structured evidence first, and put prediction / attack_probability in a separate section at the end.
Run two prompts:one with raw evidence only
one with model signal included
compare the outputs

Sometimes hide the prediction entirely and let the LLM assess from the evidence packet.
My recommendation for your current project
Keep attack_probability and risk_level for now because they make the prototype easy to understand.
But don’t present them as “the truth.”
Present them as model-derived signal.
If you want to move toward a stronger watcher agent, we should eventually reduce how much the LLM depends on the model’s own verdict.