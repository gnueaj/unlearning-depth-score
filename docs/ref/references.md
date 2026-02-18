# References (BibTeX)

Section-organized bibtex entries for the paper.
Sources: DBLP, OpenReview, Semantic Scholar. Verified 2026-02-17.

---

## Section 1: Introduction

{% raw %}
```bibtex
@inproceedings{carlini2021extracting,
  author       = {Nicholas Carlini and
                  Florian Tram{\`{e}}r and
                  Eric Wallace and
                  Matthew Jagielski and
                  Ariel Herbert{-}Voss and
                  Katherine Lee and
                  Adam Roberts and
                  Tom B. Brown and
                  Dawn Song and
                  {\'{U}}lfar Erlingsson and
                  Alina Oprea and
                  Colin Raffel},
  title        = {Extracting Training Data from Large Language Models},
  booktitle    = {30th {USENIX} Security Symposium, {USENIX} Security 2021,
                  August 11-13, 2021},
  pages        = {2633--2650},
  publisher    = {{USENIX} Association},
  year         = {2021},
  url          = {https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting},
}
% Source: DBLP conf/uss/CarliniTWJHLRBS21

@inproceedings{tirumala2022memorization,
  author       = {Kushal Tirumala and
                  Aram H. Markosyan and
                  Luke Zettlemoyer and
                  Armen Aghajanyan},
  title        = {Memorization Without Overfitting: Analyzing the Training Dynamics
                  of Large Language Models},
  booktitle    = {Advances in Neural Information Processing Systems 35: Annual Conference
                  on Neural Information Processing Systems 2022, NeurIPS 2022,
                  New Orleans, LA, USA, November 28 - December 9, 2022},
  year         = {2022},
  url          = {https://papers.nips.cc/paper_files/paper/2022/hash/fa0509f4dab6807e2cb465715bf2d249-Abstract-Conference.html},
}
% Source: DBLP conf/nips/TirumalaMZA22

@article{bengio2025aisafety,
  author       = {Yoshua Bengio and
                  S{\"{o}}ren Mindermann and
                  Daniel Privitera and
                  Tamay Besiroglu and
                  Rishi Bommasani and
                  Stephen Casper and
                  Yejin Choi and
                  others},
  title        = {International {AI} Safety Report},
  journal      = {CoRR},
  volume       = {abs/2501.17805},
  year         = {2025},
  url          = {https://arxiv.org/abs/2501.17805},
}
% Source: arXiv 2501.17805. 96 authors total; truncated with "and others".

@inproceedings{bourtoule2021machine,
  author       = {Lucas Bourtoule and
                  Varun Chandrasekaran and
                  Christopher A. Choquette{-}Choo and
                  Hengrui Jia and
                  Adelin Travers and
                  Baiwu Zhang and
                  David Lie and
                  Nicolas Papernot},
  title        = {Machine Unlearning},
  booktitle    = {42nd {IEEE} Symposium on Security and Privacy, {SP} 2021,
                  San Francisco, CA, USA, 24-27 May 2021},
  pages        = {141--159},
  publisher    = {{IEEE}},
  year         = {2021},
  url          = {https://doi.org/10.1109/SP40001.2021.00019},
  doi          = {10.1109/SP40001.2021.00019},
}
% Source: IEEE Xplore 9519428
```
{% endraw %}

---

## Section 2: Related Work — Unlearning Methods

{% raw %}
```bibtex
@inproceedings{jang2023knowledge,
  author       = {Joel Jang and
                  Dongkeun Yoon and
                  Sohee Yang and
                  Sungmin Cha and
                  Moontae Lee and
                  Lajanugen Logeswaran and
                  Minjoon Seo},
  title        = {Knowledge Unlearning for Mitigating Privacy Risks in Language Models},
  booktitle    = {Proceedings of the 61st Annual Meeting of the Association for
                  Computational Linguistics (Volume 1: Long Papers), {ACL} 2023,
                  Toronto, Canada, July 9-14, 2023},
  pages        = {14389--14408},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://doi.org/10.18653/v1/2023.acl-long.805},
  doi          = {10.18653/V1/2023.ACL-LONG.805},
}
% Source: DBLP conf/acl/JangYYCLLS23

@inproceedings{yao2024large,
  author       = {Yuanshun Yao and
                  Xiaojun Xu and
                  Yang Liu},
  title        = {Large Language Model Unlearning},
  booktitle    = {Advances in Neural Information Processing Systems 38: Annual Conference
                  on Neural Information Processing Systems 2024, NeurIPS 2024,
                  Vancouver, BC, Canada, December 10-15, 2024},
  year         = {2024},
  url          = {http://papers.nips.cc/paper_files/paper/2024/hash/be52acf6bccf4a8c0a90fe2f5cfcead3-Abstract-Conference.html},
}
% Source: DBLP conf/nips/YaoXL24

@inproceedings{zhang2024negative,
  author       = {Ruiqi Zhang and
                  Licong Lin and
                  Yu Bai and
                  Song Mei},
  title        = {Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning},
  booktitle    = {First Conference on Language Modeling, {COLM} 2024},
  year         = {2024},
  url          = {https://openreview.net/forum?id=MXLBXjQkmb},
}
% Source: OpenReview MXLBXjQkmb (COLM 2024)

@inproceedings{fan2025simplicity,
  author       = {Chongyu Fan and
                  Jiancheng Liu and
                  Licong Lin and
                  Jinghan Jia and
                  Ruiqi Zhang and
                  Song Mei and
                  Sijia Liu},
  title        = {Simplicity Prevails: Rethinking Negative Preference Optimization for {LLM} Unlearning},
  booktitle    = {Advances in Neural Information Processing Systems 38: Annual Conference
                  on Neural Information Processing Systems 2025, NeurIPS 2025},
  year         = {2025},
  url          = {https://openreview.net/forum?id=JbvSQm5h1l},
}
% Source: OpenReview JbvSQm5h1l (NeurIPS 2025 poster)

@inproceedings{maini2024tofu,
  author       = {Pratyush Maini and
                  Zhili Feng and
                  Avi Schwarzschild and
                  Zachary Chase Lipton and
                  J. Zico Kolter},
  title        = {{TOFU}: A Task of Fictitious Unlearning for {LLM}s},
  booktitle    = {First Conference on Language Modeling, {COLM} 2024},
  year         = {2024},
  url          = {https://openreview.net/forum?id=B41hNBoWLo},
}
% Source: OpenReview B41hNBoWLo (COLM 2024)

@inproceedings{dorna2025openunlearning,
  author       = {Vineeth Dorna and
                  Anmol Mekala and
                  Wenlong Zhao and
                  Andrew McCallum and
                  Zachary Chase Lipton and
                  J. Zico Kolter and
                  Pratyush Maini},
  title        = {OpenUnlearning: Accelerating {LLM} Unlearning via Unified Benchmarking
                  of Methods and Metrics},
  booktitle    = {Advances in Neural Information Processing Systems 38: Annual Conference
                  on Neural Information Processing Systems 2025, NeurIPS 2025,
                  Datasets and Benchmarks Track},
  year         = {2025},
  url          = {https://arxiv.org/abs/2506.12618},
}
% Source: arXiv 2506.12618. NeurIPS 2025 D&B track. GitHub: locuslab/open-unlearning

@article{lee2026comparator,
  author       = {Jaeung Lee and
                  Suhyeon Yu and
                  Yurim Jang and
                  Simon S. Woo and
                  Jaemin Jo},
  title        = {Unlearning Comparator: {A} Visual Analytics System for Comparative
                  Evaluation of Machine Unlearning Methods},
  journal      = {{IEEE} Transactions on Visualization and Computer Graphics},
  volume       = {32},
  number       = {3},
  pages        = {2852--2867},
  year         = {2026},
  url          = {https://doi.org/10.1109/TVCG.2026.3658325},
  doi          = {10.1109/TVCG.2026.3658325},
}
% Source: IEEE Xplore 11364307. TVCG v32(3), March 2026. IEEE VIS 2025 paper.

@inproceedings{mekala2025alternate,
  author       = {Anmol Reddy Mekala and
                  Vineeth Dorna and
                  Shreya Dubey and
                  Abhishek Lalwani and
                  David Koleczek and
                  Mukund Rungta and
                  Sadid A. Hasan and
                  Elita A. Lobo},
  title        = {Alternate Preference Optimization for Unlearning Factual Knowledge
                  in Large Language Models},
  booktitle    = {Proceedings of the 31st International Conference on Computational
                  Linguistics, {COLING} 2025, Abu Dhabi, UAE, January 19-24, 2025},
  pages        = {3732--3752},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://aclanthology.org/2025.coling-main.252/},
}
% Source: DBLP conf/coling/MekalaDDLKRHL25

@inproceedings{li2024wmdp,
  author       = {Nathaniel Li and
                  Alexander Pan and
                  Anjali Gopal and
                  Summer Yue and
                  Daniel Berrios and
                  Alice Gatti and
                  Justin D. Li and
                  Ann{-}Kathrin Dombrowski and
                  Shashwat Goel and
                  Gabriel Mukobi and
                  Nathan Helm{-}Burger and
                  Rassin Lababidi and
                  Lennart Justen and
                  Andrew B. Liu and
                  Michael Chen and
                  Isabelle Barrass and
                  Oliver Zhang and
                  Xiaoyuan Zhu and
                  Rishub Tamirisa and
                  Bhrugu Bharathi and
                  Ariel Herbert{-}Voss and
                  Cort B. Breuer and
                  Andy Zou and
                  Mantas Mazeika and
                  Zifan Wang and
                  Palash Oswal and
                  Weiran Lin and
                  Adam A. Hunt and
                  Justin Tienken{-}Harder and
                  Kevin Y. Shih and
                  Kemper Talley and
                  John Guan and
                  Ian Steneker and
                  David Campbell and
                  Brad Jokubaitis and
                  Steven Basart and
                  Stephen Fitz and
                  Ponnurangam Kumaraguru and
                  Kallol Krishna Karmakar and
                  Uday Kiran Tupakula and
                  Vijay Varadharajan and
                  Yan Shoshitaishvili and
                  Jimmy Ba and
                  Kevin M. Esvelt and
                  Alexandr Wang and
                  Dan Hendrycks},
  title        = {The {WMDP} Benchmark: Measuring and Reducing Malicious Use with Unlearning},
  booktitle    = {Forty-first International Conference on Machine Learning, {ICML} 2024,
                  Vienna, Austria, July 21-27, 2024},
  series       = {Proceedings of Machine Learning Research},
  volume       = {235},
  pages        = {28525--28550},
  publisher    = {{PMLR}},
  year         = {2024},
  url          = {https://proceedings.mlr.press/v235/li24bc.html},
}
% Source: DBLP conf/icml/LiPGYBGLDGMHLJL24

@inproceedings{dong2025undial,
  author       = {Yijiang River Dong and
                  Hongzhou Lin and
                  Mikhail Belkin and
                  Ram{\'{o}}n Huerta and
                  Ivan Vulic},
  title        = {{UNDIAL}: Self-Distillation with Adjusted Logits for Robust Unlearning
                  in Large Language Models},
  booktitle    = {Proceedings of the 2025 Conference of the Nations of the Americas
                  Chapter of the Association for Computational Linguistics: Human Language
                  Technologies, {NAACL} 2025, Albuquerque, New Mexico, USA,
                  April 29 - May 4, 2025},
  pages        = {8827--8840},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://doi.org/10.18653/v1/2025.naacl-long.444},
  doi          = {10.18653/V1/2025.NAACL-LONG.444},
}
% Source: DBLP conf/naacl/DongLBHV25

@article{eldan2023whos,
  author       = {Ronen Eldan and
                  Mark Russinovich},
  title        = {Who's Harry Potter? Approximate Unlearning in LLMs},
  journal      = {CoRR},
  volume       = {abs/2310.02238},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2310.02238},
  doi          = {10.48550/ARXIV.2310.02238},
  eprinttype   = {arXiv},
  eprint       = {2310.02238},
}
% Source: DBLP journals/corr/abs-2310-02238 (arXiv only, no published version)

@article{xu2025unlearning,
  author       = {Xiaoyu Xu and
                  Xiang Yue and
                  Yang Liu and
                  Qingqing Ye and
                  Huadi Zheng and
                  Peizhao Hu and
                  Minxin Du and
                  Haibo Hu},
  title        = {Unlearning Isn't Deletion: Investigating Reversibility of Machine
                  Unlearning in {LLMs}},
  journal      = {CoRR},
  volume       = {abs/2505.16831},
  year         = {2025},
  url          = {https://arxiv.org/abs/2505.16831},
}
% Source: DBLP journals/corr/abs-2505-16831 (arXiv only)
```
{% endraw %}

---

## Section 2: Related Work — Table 1 (White-box Analysis)

{% raw %}
```bibtex
@inproceedings{hong2024dissecting,
  author       = {Yihuai Hong and
                  Yuelin Zou and
                  Lijie Hu and
                  Ziqian Zeng and
                  Di Wang and
                  Haiqin Yang},
  title        = {Dissecting Fine-Tuning Unlearning in Large Language Models},
  booktitle    = {Proceedings of the 2024 Conference on Empirical Methods in Natural
                  Language Processing, {EMNLP} 2024, Miami, FL, USA, November 12-16,
                  2024},
  pages        = {3933--3941},
  publisher    = {Association for Computational Linguistics},
  year         = {2024},
  url          = {https://doi.org/10.18653/v1/2024.emnlp-main.228},
  doi          = {10.18653/V1/2024.EMNLP-MAIN.228},
}
% Source: DBLP conf/emnlp/HongZHZ0Y24

@inproceedings{hong2025intrinsic,
  author       = {Yihuai Hong and
                  Lei Yu and
                  Haiqin Yang and
                  Shauli Ravfogel and
                  Mor Geva},
  title        = {Intrinsic Test of Unlearning Using Parametric Knowledge Traces},
  booktitle    = {Proceedings of the 2025 Conference on Empirical Methods in Natural
                  Language Processing, {EMNLP} 2025, Suzhou, China, November 4-9,
                  2025},
  pages        = {19513--19535},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://doi.org/10.18653/v1/2025.emnlp-main.985},
  doi          = {10.18653/V1/2025.EMNLP-MAIN.985},
}
% Source: DBLP conf/emnlp/HongYYRG25

@article{lynch2024eight,
  author       = {Aengus Lynch and
                  Phillip Guo and
                  Aidan Ewart and
                  Stephen Casper and
                  Dylan Hadfield{-}Menell},
  title        = {Eight Methods to Evaluate Robust Unlearning in LLMs},
  journal      = {CoRR},
  volume       = {abs/2402.16835},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2402.16835},
  doi          = {10.48550/ARXIV.2402.16835},
  eprinttype   = {arXiv},
  eprint       = {2402.16835},
}
% Source: DBLP journals/corr/abs-2402-16835 (arXiv only)

@inproceedings{guo2025mechanistic,
  author       = {Phillip Guo and
                  Aaquib Syed and
                  Abhay Sheshadri and
                  Aidan Ewart and
                  Gintare Karolina Dziugaite},
  title        = {Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via
                  Mechanistic Localization},
  booktitle    = {Forty-second International Conference on Machine Learning, {ICML}
                  2025, Vancouver, BC, Canada, July 13-19, 2025},
  series       = {Proceedings of Machine Learning Research},
  volume       = {267},
  publisher    = {{PMLR}},
  year         = {2025},
  url          = {https://proceedings.mlr.press/v267/guo25k.html},
}
% Source: DBLP conf/icml/GuoSSED25

@inproceedings{patil2024can,
  author       = {Vaidehi Patil and
                  Peter Hase and
                  Mohit Bansal},
  title        = {Can Sensitive Information Be Deleted From LLMs? Objectives for Defending
                  Against Extraction Attacks},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=7erlRDoaV8},
}
% Source: DBLP conf/iclr/PatilHB24
```
{% endraw %}

---

## Section 2: Related Work — Representation Analysis

{% raw %}
```bibtex
@inproceedings{kornblith2019similarity,
  author       = {Simon Kornblith and
                  Mohammad Norouzi and
                  Honglak Lee and
                  Geoffrey E. Hinton},
  title        = {Similarity of Neural Network Representations Revisited},
  booktitle    = {Proceedings of the 36th International Conference on Machine Learning,
                  {ICML} 2019, 9-15 June 2019, Long Beach, California, {USA}},
  series       = {Proceedings of Machine Learning Research},
  volume       = {97},
  pages        = {3519--3529},
  publisher    = {{PMLR}},
  year         = {2019},
  url          = {http://proceedings.mlr.press/v97/kornblith19a.html},
}
% Source: DBLP conf/icml/Kornblith0LH19

@misc{nostalgebraist2020logitlens,
  author       = {nostalgebraist},
  title        = {Interpreting {GPT}: the Logit Lens},
  year         = {2020},
  howpublished = {LessWrong},
  url          = {https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens},
}
% Source: LessWrong blog post (no formal publication)

@article{kirkpatrick2017overcoming,
  author       = {James Kirkpatrick and
                  Razvan Pascanu and
                  Neil C. Rabinowitz and
                  Joel Veness and
                  Guillaume Desjardins and
                  Andrei A. Rusu and
                  Kieran Milan and
                  John Quan and
                  Tiago Ramalho and
                  Agnieszka Grabska-Barwinska and
                  Demis Hassabis and
                  Claudia Clopath and
                  Dharshan Kumaran and
                  Raia Hadsell},
  title        = {Overcoming catastrophic forgetting in neural networks},
  journal      = {Proceedings of the National Academy of Sciences},
  volume       = {114},
  number       = {13},
  pages        = {3521--3526},
  year         = {2017},
  doi          = {10.1073/pnas.1611835114},
}
% Source: DBLP + Semantic Scholar DOI verification

@inproceedings{raghu2017svcca,
  author       = {Maithra Raghu and
                  Justin Gilmer and
                  Jason Yosinski and
                  Jascha Sohl-Dickstein},
  title        = {{SVCCA}: Singular Vector Canonical Correlation Analysis for Deep Learning
                  Dynamics and Interpretability},
  booktitle    = {Advances in Neural Information Processing Systems 30: Annual Conference
                  on Neural Information Processing Systems 2017, December 4-9, 2017,
                  Long Beach, CA, {USA}},
  pages        = {6076--6085},
  year         = {2017},
  url          = {https://proceedings.neurips.cc/paper/2017/hash/dc6a7e655d7e5840e66733e9ee67cc69-Abstract.html},
}
% Source: DBLP conf/nips/RaghuGYS17

@inproceedings{meng2022locating,
  author       = {Kevin Meng and
                  David Bau and
                  Alex Andonian and
                  Yonatan Belinkov},
  title        = {Locating and Editing Factual Associations in {GPT}},
  booktitle    = {Advances in Neural Information Processing Systems 35: Annual Conference
                  on Neural Information Processing Systems 2022, NeurIPS 2022,
                  New Orleans, LA, USA, November 28 - December 9, 2022},
  year         = {2022},
  url          = {http://papers.nips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html},
}
% Source: DBLP conf/nips/MengBAB22

@inproceedings{ghandeharioun2024patchscopes,
  author       = {Asma Ghandeharioun and
                  Avi Caciularu and
                  Adam Pearce and
                  Lucas Dixon and
                  Mor Geva},
  title        = {Patchscopes: {A} Unifying Framework for Inspecting Hidden Representations
                  of Language Models},
  booktitle    = {Forty-first International Conference on Machine Learning, {ICML} 2024,
                  Vienna, Austria, July 21-27, 2024},
  series       = {Proceedings of Machine Learning Research},
  volume       = {235},
  pages        = {15466--15490},
  publisher    = {{PMLR}},
  year         = {2024},
  url          = {https://proceedings.mlr.press/v235/ghandeharioun24a.html},
}
% Source: DBLP conf/icml/GhandehariounCP24
```
{% endraw %}

---

## Section 4: Evaluation — MIA & Privacy Metrics

{% raw %}
```bibtex
@inproceedings{shokri2017membership,
  author       = {Reza Shokri and
                  Marco Stronati and
                  Congzheng Song and
                  Vitaly Shmatikov},
  title        = {Membership Inference Attacks Against Machine Learning Models},
  booktitle    = {2017 {IEEE} Symposium on Security and Privacy, {SP} 2017,
                  San Jose, CA, USA, May 22-26, 2017},
  pages        = {3--18},
  publisher    = {{IEEE} Computer Society},
  year         = {2017},
  url          = {https://doi.org/10.1109/SP.2017.41},
  doi          = {10.1109/SP.2017.41},
}
% Source: DBLP conf/sp/ShokriSSS17

@inproceedings{yeom2018privacy,
  author       = {Samuel Yeom and
                  Irene Giacomelli and
                  Matt Fredrikson and
                  Somesh Jha},
  title        = {Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting},
  booktitle    = {31st {IEEE} Computer Security Foundations Symposium, {CSF} 2018,
                  Oxford, United Kingdom, July 9-12, 2018},
  pages        = {268--282},
  publisher    = {{IEEE} Computer Society},
  year         = {2018},
  url          = {https://doi.org/10.1109/CSF.2018.00027},
  doi          = {10.1109/CSF.2018.00027},
}
% Source: DBLP conf/csfw/YeomGFJ18

@inproceedings{shi2024detecting,
  author       = {Weijia Shi and
                  Anirudh Ajith and
                  Mengzhou Xia and
                  Yangsibo Huang and
                  Daogao Liu and
                  Terra Blevins and
                  Danqi Chen and
                  Luke Zettlemoyer},
  title        = {Detecting Pretraining Data from Large Language Models},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=zWqr3MQuNs},
}
% Source: DBLP conf/iclr/ShiAXHLB0Z24

@inproceedings{duan2024membership,
  author       = {Michael Duan and
                  Anshuman Suri and
                  Niloofar Mireshghallah and
                  Sewon Min and
                  Weijia Shi and
                  Luke Zettlemoyer and
                  Yulia Tsvetkov and
                  Yejin Choi and
                  David Evans and
                  Hannaneh Hajishirzi},
  title        = {Do Membership Inference Attacks Work on Large Language Models?},
  booktitle    = {First Conference on Language Modeling, {COLM} 2024},
  year         = {2024},
  url          = {https://openreview.net/forum?id=av0D19pSkU},
}
% Source: OpenReview av0D19pSkU (COLM 2024)

@inproceedings{zhang2025minkpp,
  author       = {Jingyang Zhang and
                  Jingwei Sun and
                  Eric Yeats and
                  Yang Ouyang and
                  Martin Kuo and
                  Jianyi Zhang and
                  Hao Frank Yang and
                  Hai Li},
  title        = {Min-K\%++: Improved Baseline for Pre-Training Data Detection from
                  Large Language Models},
  booktitle    = {The Thirteenth International Conference on Learning Representations,
                  {ICLR} 2025},
  year         = {2025},
  url          = {https://openreview.net/forum?id=ZGkfoufDaU},
}
% Source: OpenReview ZGkfoufDaU (ICLR 2025 Spotlight). Previous ID hAgt0gMXjg was invalid.

@inproceedings{shi2025muse,
  author       = {Weijia Shi and
                  Jaechan Lee and
                  Yangsibo Huang and
                  Sadhika Malladi and
                  Jieyu Zhao and
                  Ari Holtzman and
                  Daogao Liu and
                  Luke Zettlemoyer and
                  Noah A. Smith and
                  Chiyuan Zhang},
  title        = {{MUSE}: Machine Unlearning Six-Way Evaluation for Language Models},
  booktitle    = {The Thirteenth International Conference on Learning Representations,
                  {ICLR} 2025},
  year         = {2025},
  url          = {https://openreview.net/forum?id=TArmA033BU},
}
% Source: DBLP conf/iclr/ShiLHMZHLZSZ25. Previous ID bGMKsMflBn was invalid.
```
{% endraw %}

---

## Section 5: Discussion & Related Expansions

{% raw %}
```bibtex
@inproceedings{liu2024revisiting,
  author       = {Yujian Liu and
                  Yang Zhang and
                  Tommi S. Jaakkola and
                  Shiyu Chang},
  title        = {Revisiting Who's Harry Potter: Towards Targeted Unlearning from a
                  Causal Intervention Perspective},
  booktitle    = {Proceedings of the 2024 Conference on Empirical Methods in Natural
                  Language Processing, {EMNLP} 2024, Miami, FL, USA, November 12-16,
                  2024},
  pages        = {8708--8731},
  publisher    = {Association for Computational Linguistics},
  year         = {2024},
  url          = {https://doi.org/10.18653/v1/2024.emnlp-main.495},
  doi          = {10.18653/V1/2024.EMNLP-MAIN.495},
}
% Source: DBLP conf/emnlp/LiuZJC24

@inproceedings{lo2024relearn,
  author       = {Michelle Lo and
                  Fazl Barez and
                  Shay B. Cohen},
  title        = {Large Language Models Relearn Removed Concepts},
  booktitle    = {Findings of the Association for Computational Linguistics, {ACL} 2024,
                  Bangkok, Thailand and virtual meeting, August 11-16, 2024},
  series       = {Findings of {ACL}},
  volume       = {{ACL} 2024},
  pages        = {8306--8323},
  publisher    = {Association for Computational Linguistics},
  year         = {2024},
  url          = {https://doi.org/10.18653/v1/2024.findings-acl.492},
  doi          = {10.18653/V1/2024.FINDINGS-ACL.492},
}
% Source: DBLP conf/acl/LoBC24

@inproceedings{hou2025decoupling,
  author       = {Lishuai Hou and
                  Zixiong Wang and
                  Gaoyang Liu and
                  Chen Wang and
                  Wei Liu and
                  Kai Peng},
  title        = {Decoupling Memories, Muting Neurons: Towards Practical Machine Unlearning
                  for Large Language Models},
  booktitle    = {Findings of the Association for Computational Linguistics, {ACL} 2025,
                  Vienna, Austria, July 27 - August 1, 2025},
  series       = {Findings of {ACL}},
  volume       = {{ACL} 2025},
  pages        = {13978--13999},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://aclanthology.org/2025.findings-acl.719/},
}
% Source: DBLP conf/acl/HouWLWL025

@inproceedings{wang2025reasoning,
  author       = {Changsheng Wang and
                  Chongyu Fan and
                  Yihua Zhang and
                  Jinghan Jia and
                  Dennis Wei and
                  Parikshit Ram and
                  Nathalie Baracaldo and
                  Sijia Liu},
  title        = {Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While
                  Preserving Reasoning Skills},
  booktitle    = {Proceedings of the 2025 Conference on Empirical Methods in Natural
                  Language Processing, {EMNLP} 2025, Suzhou, China, November 4-9, 2025},
  pages        = {4427--4443},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://doi.org/10.18653/v1/2025.emnlp-main.220},
  doi          = {10.18653/V1/2025.EMNLP-MAIN.220},
}
% Source: DBLP conf/emnlp/WangFZJWRBL25
```
{% endraw %}

---

## Verification Notes

| # | Citation Key | Venue | Source | Notes |
|---|---|---|---|---|
| 1 | `carlini2021extracting` | USENIX Security 2021 | DBLP | pp. 2633-2650 |
| 2 | `tirumala2022memorization` | NeurIPS 2022 | DBLP | |
| 3 | `jang2023knowledge` | ACL 2023 | DBLP | pp. 14389-14408 |
| 4 | `zhang2024negative` | COLM 2024 | OpenReview | Not yet on DBLP |
| 5 | `fan2025simplicity` | NeurIPS 2025 | OpenReview | Not yet on DBLP |
| 6 | `maini2024tofu` | COLM 2024 | OpenReview | Not yet on DBLP |
| 7 | `mekala2025alternate` | COLING 2025 | DBLP | pp. 3732-3752 |
| 8 | `li2024wmdp` | ICML 2024 | DBLP | PMLR v235, pp. 28525-28550 |
| 9 | `dong2025undial` | NAACL 2025 | DBLP | pp. 8827-8840 |
| 10 | `eldan2023whos` | arXiv 2023 | DBLP | No published version found |
| 11 | `hong2024dissecting` | EMNLP 2024 | DBLP | pp. 3933-3941 |
| 12 | `hong2025intrinsic` | EMNLP 2025 | DBLP | pp. 19513-19535 |
| 13 | `lynch2024eight` | arXiv 2024 | DBLP | No published version found |
| 14 | `guo2025mechanistic` | ICML 2025 | DBLP | PMLR v267 |
| 15 | `patil2024can` | ICLR 2024 | DBLP | Key matches table1.tex citeyear |
| 16 | `kornblith2019similarity` | ICML 2019 | DBLP | PMLR v97, pp. 3519-3529 |
| 17 | `nostalgebraist2020logitlens` | LessWrong 2020 | Manual | Blog post |
| 18 | `kirkpatrick2017overcoming` | PNAS 2017 | DBLP+S2 | v114(13), pp. 3521-3526 |
| 19 | `raghu2017svcca` | NeurIPS 2017 | DBLP | pp. 6076-6085 |
| 20 | `meng2022locating` | NeurIPS 2022 | DBLP | |
| 21 | `ghandeharioun2024patchscopes` | ICML 2024 | DBLP | PMLR v235, pp. 15466-15490 |
| 22 | `shokri2017membership` | IEEE S&P 2017 | DBLP | pp. 3-18 |
| 23 | `yeom2018privacy` | IEEE CSF 2018 | DBLP | pp. 268-282 |
| 24 | `shi2024detecting` | ICLR 2024 | DBLP | |
| 25 | `duan2024membership` | COLM 2024 | OpenReview | Not yet on DBLP |
| 26 | `zhang2025minkpp` | ICLR 2025 | OpenReview | Spotlight. ID=ZGkfoufDaU. Authors corrected |
| 27 | `shi2025muse` | ICLR 2025 | DBLP | |
| 28 | `liu2024revisiting` | EMNLP 2024 | DBLP | pp. 8708-8731 |
| 29 | `lo2024relearn` | Findings ACL 2024 | DBLP | pp. 8306-8323 |
| 30 | `hou2025decoupling` | Findings ACL 2025 | DBLP | pp. 13978-13999 |
| 31 | `wang2025reasoning` | EMNLP 2025 | DBLP | pp. 4427-4443 |
| 32 | `bengio2025aisafety` | arXiv 2025 | arXiv | 96 authors, truncated with "and others" |
| 33 | `bourtoule2021machine` | IEEE S&P 2021 | DBLP | pp. 141-159 |
| 34 | `dorna2025openunlearning` | NeurIPS 2025 D&B | arXiv | COLM not on DBLP; arXiv 2506.12618 |
| 35 | `lee2026comparator` | IEEE TVCG 2026 | IEEE Xplore | v32(3), pp. 2852-2867, DOI:10.1109/TVCG.2026.3658325 |
| 36 | `yao2024large` | NeurIPS 2024 | DBLP | conf/nips/YaoXL24. arXiv preprint was 2310.10683 (2023) |
| 37 | `fan2025simplicity` | NeurIPS 2025 | OpenReview | poster. ID=JbvSQm5h1l |
