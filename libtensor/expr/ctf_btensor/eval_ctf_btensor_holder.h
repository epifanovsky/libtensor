#ifndef LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_HOLDER_H
#define LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_HOLDER_H

#include <libutil/singleton.h>
#include <libtensor/expr/eval/eval_register.h>
#include "eval_ctf_btensor.h"

namespace libtensor {
namespace expr {


/** \brief Holder for the global ctf_btensor expression evaluator

    \ingroup libtensor_expr_ctf_btensor
 **/
template<typename T>
class eval_ctf_btensor_holder :
    public libutil::singleton< eval_ctf_btensor_holder<T> > {

    friend class libutil::singleton< eval_ctf_btensor_holder<T> >;

private:
    size_t m_cnt;
    eval_ctf_btensor<T> m_eval; //!< Evaluator

public:
    void inc_counter() {
        if(m_cnt == 0) {
            eval_register::get_instance().add_evaluator(m_eval);
        }
        m_cnt++;
    }

    void dec_counter() {
        if(m_cnt > 0) m_cnt--;
        if(m_cnt == 0) {
            eval_register::get_instance().remove_evaluator(m_eval);
        }
    }

protected:
    /** \brief Protected constructor
     **/
    eval_ctf_btensor_holder() : m_cnt(0) { }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_HOLDER_H
