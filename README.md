# SMAttack
- Skeleton-based action recognition has made remarkable strides in applications such as intelligent surveillance, sports analysis, and rehabilitation assessment. 
However, deep learning models in this field are highly vulnerable to adversarial attacks, which leads to severe security risks, including system failures and misclassification of critical actions. 
Existing attack methods primarily rely on non-manifold perturbations that enforce strict surface-level similarity between adversarial and original samples. 
These approaches overlook the natural variations in human motion, making them easier to detect and limiting their transferability. 
Moreover, rigid similarity constraints often contradict real-world movement dynamics, reducing the feasibility of such attacks.
To address these challenges, we propose the \textbf{Skeleton Manifold Attack~(SMAttack)}, a novel adversarial attack framework that operates in the frequency domain. 
Instead of directly perturbing skeleton sequences in the spatial domain, SMAttack projects them into the spectral space and injects adversarial perturbations into their amplitude components. 
This enables the generation of adversarial samples that better align with the natural distribution of human motion while maintaining high attack effectiveness. 
To further refine the motion consistency, we introduce a \textit{pose smoothing mechanism} that reduces abrupt distortions and enhances the realism of adversarial sequences.
Extensive experiments on benchmark datasets demonstrate that SMAttack significantly outperforms conventional non-manifold attacks in both transferability and undefendabality. 

> python3.9 Attack.py --config ./config/stgcn-ntu60-cs.yaml

