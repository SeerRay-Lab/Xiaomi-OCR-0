We then define the following variables: \(\mathbf{P}_i^j = \prod_{t=i}^j (\mathbf{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top) \in \mathbb{R}^{d \times d}\), \(\mathbf{H}_i^j = \sum_{t=i}^j \beta_t (\boldsymbol{v}_t \boldsymbol{k}_t^\top) \mathbf{P}_{t+1}^j \in \mathbb{R}^{d \times d}\), where we let \(\mathbf{P}_i^j = \mathbf{I}\) whenever \(i > j\). Intuitively, \(\mathbf{P}_i^j\) is the "decay factor" to be applied to \(\mathbf{S}_i\) for obtaining \(\mathbf{S}_j\), and \(\mathbf{H}_i^j\) represents the contributions to \(\mathbf{S}_j\) starting from token \(i\). (Hence \(\mathbf{S}_t = \mathbf{H}_1^t\)). The chunkwise recurrence can then be written as,

\[\mathbf {S} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {0} \mathbf {P} _ {[ t ]} ^ {r} + \mathbf {H} _ {[ t ]} ^ {r} \tag{5}\]

where we define the chunkwise variables \(\mathbf{S}_{[t]}^{i} = \mathbf{S}_{tC + i}\), \(\mathbf{P}_{[t]}^{r} = \mathbf{P}_{tC + 1}^{tC + r}\), \(\mathbf{H}_{[t]}^{r} = \mathbf{H}_{tC + 1}^{tC + r}\). Here we have \(\frac{L}{C}\) chunks of size \(C\). The trick is to now efficiently represent the \(\mathbf{P}_{[t]}^{r}, \mathbf{H}_{[t]}^{r} \in \mathbb{R}^{d \times d}\) matrices using a similar approach described in §3.1, so that these matrices can be stored in \(\mathcal{O}(d)\) memory,

\[\mathbf {P} _ {[ t ]} ^ {r} = \mathbf {I} - \sum_ {i = 1} ^ {r} \boldsymbol {w} _ {[ t ]} ^ {i} \boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}}, \quad \mathbf {H} _ {[ t ]} ^ {r} = \sum_ {i = 1} ^ {r} \boldsymbol {u} _ {[ t ]} ^ {i} \boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \quad \in \mathbb {R} ^ {d \times d} \tag{6}\]

\[\boldsymbol {w} _ {[ t ]} ^ {r} = \beta_ {[ t ]} ^ {r} \left(\boldsymbol {k} _ {[ t ]} ^ {r} - \sum_ {i = 1} ^ {r - 1} \boldsymbol {w} _ {[ t ]} ^ {i} (\boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \boldsymbol {k} _ {[ t ]} ^ {r})\right), \quad \boldsymbol {u} _ {[ t ]} ^ {r} = \beta_ {[ t ]} ^ {r} \left(\boldsymbol {v} _ {[ t ]} ^ {r} - \sum_ {i = 1} ^ {r - 1} \boldsymbol {u} _ {[ t ]} ^ {i} (\boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \boldsymbol {k} _ {[ t ]} ^ {r})\right) \quad \in \mathbb {R} ^ {d} \tag{7}\]

The derivations for the above can be found in the appendix. Subsequently, based on Eq. 5, we can obtain the chunk-level recurrence for hidden states and outputs as,

\[\mathbf {S} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {0} - \left(\mathbf {S} _ {[ t ]} ^ {0} \sum_ {i = 1} ^ {r} \boldsymbol {w} _ {[ t ]} ^ {i} \boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}}\right) + \sum_ {i = 1} ^ {r} \boldsymbol {u} _ {[ t ]} ^ {i} \boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} = \mathbf {S} _ {[ t ]} ^ {0} + \sum_ {i = 1} ^ {r} \left(\boldsymbol {u} _ {[ t ]} ^ {i} - \mathbf {S} _ {[ t ]} ^ {0} \boldsymbol {w} _ {[ t ]} ^ {i}\right) \boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}},\]

\[\boldsymbol {o} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {r} \boldsymbol {q} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {0} \boldsymbol {q} _ {[ t ]} ^ {r} + \sum_ {i = 1} ^ {r} \left(\boldsymbol {u} _ {[ t ]} ^ {i} - \mathbf {S} _ {[ t ]} ^ {0} \boldsymbol {w} _ {[ t ]} ^ {i}\right) \left(\boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \boldsymbol {q} _ {[ t ]} ^ {i}\right).\]

Letting \(\mathbf{S}_{[t]} = \mathbf{S}_{[t]}^{0}\), the above can be simplified to matrix notations similarly to Eq.1-2,

\[\mathbf {S} _ {[ t + 1 ]} = \mathbf {S} _ {[ t ]} + \left(\mathbf {U} _ {[ t ]} - \mathbf {W} _ {[ t ]} \mathbf {S} _ {[ t ]} ^ {\top}\right) ^ {\top} \mathbf {K} _ {[ t ]}, \tag{8}\]

\[\mathbf {O} _ {[ t ]} = \mathbf {Q} _ {[ t ]} \mathbf {S} _ {[ t ]} ^ {\top} + (\mathbf {Q} _ {[ t ]} \mathbf {K} _ {[ t ]} ^ {\top} \odot \mathbf {M}) \left(\mathbf {U} _ {[ t ]} - \mathbf {W} _ {[ t ]} \mathbf {S} _ {[ t ]} ^ {\top}\right) \tag{9}\]

where \(\square_{[t]} = \square_{[t]}^{1:C} \in \mathbb{R}^{C \times d}\) for \(\square \in \{\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O}, \mathbf{U}, \mathbf{W}\}\) defines the chunkwise matrices that are formed from stacking the \(q_t, k_t, v_t, o_t, u_t, w_t\) vectors.

Practical considerations. In the above, Eq. 7 is fully recurrent and thus cannot use tensor cores written as is. To solve this, we further leverage the UT transform [44, 23] (see §B.2 for derivations):

\[\mathbf {T} _ {[ t ]} = \left(\mathbf {I} + \operatorname{tril} (\operatorname{diag} (\beta_ {[ t ]}) \mathbf {K} _ {[ t ]} \mathbf {K} _ {[ t ]} ^ {\top}, - 1)\right) ^ {- 1} \operatorname{diag} \left(\beta_ {[ t ]}\right) \tag{10}\]

\[\mathbf {W} _ {[ t ]} = \mathbf {T} _ {[ t ]} \mathbf {K} _ {[ t ]}, \quad \mathbf {U} _ {[ t ]} = \mathbf {T} _ {[ t ]} \mathbf {V} _ {[ t ]} \tag{11}
