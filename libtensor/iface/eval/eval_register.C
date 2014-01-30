#include <algorithm>
#include "eval_register.h"

namespace libtensor {
namespace iface {


void eval_register::add_evaluator(const eval_i &e) {

    m_eval.push_back(&e);
}


void eval_register::remove_evaluator(const eval_i &e) {

    std::vector<const eval_i*>::iterator i =
        std::find(m_eval.begin(), m_eval.end(), &e);
    if(i == m_eval.end()) return;
    m_eval.erase(i);
}


void eval_register::try_evaluators(eval_selector_i &es) {

    for(std::vector<const eval_i*>::const_iterator i = m_eval.begin();
        i != m_eval.end(); ++i) es.try_evaluator(**i);
}


} // namespace iface
} // namespace libtensor
