#include <libtensor/expr/expr_exception.h>
#include "default_eval_selector.h"

namespace libtensor {
namespace expr {


const char default_eval_selector::k_clazz[] = "expr::default_eval_selector";


default_eval_selector::~default_eval_selector() {

}


const eval_i &default_eval_selector::get_selected() const {

    static const char method[] = "get_selected()";

    if(!m_eval) {
        throw expr_exception("libtensor", k_clazz, method, __FILE__, __LINE__,
            "Unable to find a suitable evaluator for the expression");
    }
    return *m_eval;
}


void default_eval_selector::try_evaluator(const eval_i &e) {

    if(m_eval) return;
    if(e.can_evaluate(m_expr)) m_eval = &e;
}


} // namespace expr
} // namespace libtensor
