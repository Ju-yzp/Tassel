#ifndef TASSEL_CORE_LM_OPTIMIZER_H_
#define TASSEL_CORE_LM_OPTIMIZER_H_

namespace tassel_core {
class LMOptimizer {
public:
    LMOptimizer(int max_itemrator_num, int thread_num);
    ~LMOptimizer();

private:
    int max_itemrator_num_;

    int thread_num_;
};
}  // namespace tassel_core
#endif /* TASSEL_CORE_LM_OPTIMIZER_H_ */
