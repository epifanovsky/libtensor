#include "default_eval_selector.h"

namespace libtensor {
namespace expr {


default_eval_selector::~default_eval_selector() {

}


const eval_i &default_eval_selector::get_selected() const {

    if(!m_eval) throw 0;
    return *m_eval;
}


void default_eval_selector::try_evaluator(const eval_i &e) {

    if(m_eval) return;
    if(e.can_evaluate(m_expr)) m_eval = &e;
}


} // namespace expr
} // namespace libtensor
