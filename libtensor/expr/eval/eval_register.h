#ifndef LIBTENSOR_EXPR_EVAL_REGISTER_H
#define LIBTENSOR_EXPR_EVAL_REGISTER_H

#include <vector>
#include <libutil/singleton.h>
#include "eval_selector_i.h"

namespace libtensor {
namespace expr {


/** \brief Keeps records of all expression evaluators currently available
        for use

    \ingroup libtensor_iface
 **/
class eval_register : public libutil::singleton<eval_register> {
    friend class libutil::singleton<eval_register>;

private:
    std::vector<const eval_i*> m_eval; //!< List of expression evaluators

public:
    /** \brief Adds an evaluator to the list
     **/
    void add_evaluator(const eval_i &e);

    /** \brief Removes an evaluator from the list
     **/
    void remove_evaluator(const eval_i &e);

    /** \brief Submits all registered evaluators for trial
     **/
    void try_evaluators(eval_selector_i &es);

protected:
    /** \brief Protected constructor
     **/
    eval_register() { }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_REGISTER_H
