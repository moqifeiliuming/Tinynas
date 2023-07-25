from .builder import BUDGETS
from .base import BudgetBase

# 该模块主要是比较某一型号的实际值和预算值，仅仅实现了ExceedBudget类，如果模型值超过config中设置的预算值，则返回false
@BUDGETS.register_module(module_name = 'layers')    # 网络的Conv层预算
@BUDGETS.register_module(module_name = 'model_size')  # 网络参数预算的数量
@BUDGETS.register_module(module_name = 'flops')  # 网络的FLOPs预算
@BUDGETS.register_module(module_name = 'latency')  # 网络的延迟预算
@BUDGETS.register_module(module_name = 'max_feature')  # 网络MCU的最大特征图预算
@BUDGETS.register_module(module_name = 'efficient_score') # 网络的有效分数预算
class ExceedBudget(BudgetBase):
    def __init__(self, name, budget, logger, **kwargs):
        super().__init__(name, budget)
        self.logger = logger 

    def compare(self, model_info):
        input  = model_info[self.name]
        if self.budget < input:
            self.logger.debug(
                '{} value = {} in the structure_info exceed the budget ={}'.
                format(self.name, input, self.budget))
            return False
        return True

    def __call__(self, model_info):
        return self.compare(model_info)
