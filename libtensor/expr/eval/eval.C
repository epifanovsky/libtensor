#include "default_eval_selector.h"
#include "eval.h"
#include "eval_register.h"

namespace libtensor {
namespace expr {


void eval::evaluate(const expr_tree &e, eval_selector_i &es) const {

    eval_register::get_instance().try_evaluators(es);
    es.get_selected().evaluate(e);
}


void eval::evaluate(const expr_tree &e) const {

    default_eval_selector es(e);
    eval_register::get_instance().try_evaluators(es);
    es.get_selected().evaluate(e);
}


} // namespace expr
} // namespace libtensor
