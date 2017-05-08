/*******************************************************************\

   Module: Symbolic Execution

   Author: Daniel Kroening, kroening@kroening.com Lucas Cordeiro,
     lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>
#include <assert.h>
#include <iostream>
#include <vector>

#include <prefix.h>
#include <std_expr.h>
#include <expr_util.h>

#include "goto_symex.h"
#include "goto_symex_state.h"
#include "execution_state.h"
#include "symex_target_equation.h"
#include "reachability_tree.h"

#include <std_expr.h>
#include "../ansi-c/c_types.h"
#include <simplify_expr.h>
#include "config.h"

void
goto_symext::claim(const expr2tc &claim_expr, const std::string &msg) {

  if (unwinding_recursion_assumption)
    return ;

  // Can happen when evaluating certain special intrinsics. Gulp.
  if (cur_state->guard.is_false())
    return;

  total_claims++;

  expr2tc new_expr = claim_expr;
  cur_state->rename(new_expr);

  // first try simplifier on it
  do_simplify(new_expr);

  if (is_true(new_expr))
    return;

  cur_state->guard.guard_expr(new_expr);
  cur_state->global_guard.guard_expr(new_expr);
  remaining_claims++;
  target->assertion(cur_state->guard.as_expr(), new_expr, msg,
                    cur_state->gen_stack_trace(),
                    cur_state->source);
}

void
goto_symext::assume(const expr2tc &assumption)
{

  // Irritatingly, assumption destroys its expr argument
  expr2tc tmp_guard = cur_state->guard.as_expr();
  cur_state->global_guard.guard_expr(tmp_guard);
  target->assumption(tmp_guard, assumption, cur_state->source);
  return;
}

goto_symext::symex_resultt *
goto_symext::get_symex_result(void)
{

  return new goto_symext::symex_resultt(target, total_claims, remaining_claims);
}

void
goto_symext::symex_step(reachability_treet & art)
{

  assert(!cur_state->call_stack.empty());

  const goto_programt::instructiont &instruction = *cur_state->source.pc;

  // depth exceeded?
  {
    if (depth_limit != 0 && cur_state->depth > depth_limit)
      cur_state->guard.add(false_expr);
    cur_state->depth++;
  }

  // actually do instruction
  switch (instruction.type) {
  case SKIP:
  case LOCATION:
    // really ignore
    cur_state->source.pc++;
    break;

  case END_FUNCTION:
    symex_end_of_function();

    // Potentially skip to run another function ptr target; if not,
    // continue
    if (!run_next_function_ptr_target(false))
      cur_state->source.pc++;
    break;

  case GOTO:
  {
    expr2tc tmp(instruction.guard);
    replace_nondet(tmp);

    dereference(tmp, false);
    replace_dynamic_allocation(tmp);

    symex_goto(tmp);
  }
  break;

  case ASSUME:
    if (!cur_state->guard.is_false()) {
      expr2tc tmp = instruction.guard;
      replace_nondet(tmp);

      dereference(tmp, false);
      replace_dynamic_allocation(tmp);

      cur_state->rename(tmp);
      do_simplify(tmp);

      if (!is_true(tmp)) {
        expr2tc tmp2 = tmp;
        expr2tc tmp3 = tmp2;
        cur_state->guard.guard_expr(tmp2);

        assume(tmp2);

        // we also add it to the state guard
        cur_state->guard.add(tmp3);
      }
    }
    cur_state->source.pc++;
    break;

  case ASSERT:
    if (!cur_state->guard.is_false()) {
      if (!no_assertions ||
        !cur_state->source.pc->location.user_provided()
        || deadlock_check) {

        std::string msg = cur_state->source.pc->location.comment().as_string();
        if (msg == "") msg = "assertion";

        expr2tc tmp = instruction.guard;
        replace_nondet(tmp);

        dereference(tmp, false);
        replace_dynamic_allocation(tmp);

        claim(tmp, msg);
      }
    }
    cur_state->source.pc++;
    break;

  case RETURN:
    if (!cur_state->guard.is_false()) {
      expr2tc thecode = instruction.code, assign;
      if (make_return_assignment(assign, thecode)) {
        goto_symext::symex_assign(assign);
      }

      symex_return();
    }

    cur_state->source.pc++;
    break;

  case ASSIGN:
    if (!cur_state->guard.is_false()) {
      code_assign2tc deref_code = instruction.code;

      // XXX jmorse -- this is not fully symbolic.
      if (thrown_obj_map.find(cur_state->source.pc) != thrown_obj_map.end()) {
        symbol2tc thrown_obj = thrown_obj_map[cur_state->source.pc];

        if (is_pointer_type(deref_code.get()->target.get()->type)
            && !is_pointer_type(thrown_obj.get()->type))
        {
          expr2tc new_thrown_obj(new address_of2t(thrown_obj.get()->type, thrown_obj));
          deref_code.get()->source = new_thrown_obj;
        }
        else
          deref_code.get()->source = thrown_obj;

        thrown_obj_map.erase(cur_state->source.pc);
      }

      replace_nondet(deref_code);

      code_assign2t &assign = to_code_assign2t(deref_code);

      dereference(assign.target, true);
      dereference(assign.source, false);
      replace_dynamic_allocation(deref_code);

      symex_assign(deref_code);
    }

    cur_state->source.pc++;
    break;

  case FUNCTION_CALL:
  {
    expr2tc deref_code = instruction.code;
    replace_nondet(deref_code);

    code_function_call2t &call = to_code_function_call2t(deref_code);

    if (!is_nil_expr(call.ret)) {
      dereference(call.ret, true);
    }

    replace_dynamic_allocation(deref_code);

    for (std::vector<expr2tc>::iterator it = call.operands.begin();
         it != call.operands.end(); it++)
      if (!is_nil_expr(*it))
        dereference(*it, false);

    // Always run intrinsics, whether guard is false or not. This is due to the
    // unfortunate circumstance where a thread starts with false guard due to
    // decision taken in another thread in this trace. In that case the
    // terminate intrinsic _has_ to run, or we explode.
    if (is_symbol2t(call.function)) {
      const irep_idt &id = to_symbol2t(call.function).thename;
      if (has_prefix(id.as_string(), "c::__ESBMC")) {
        cur_state->source.pc++;
        std::string name = id.as_string().substr(3);
        run_intrinsic(call, art, name);
        return;
      } else if (has_prefix(id.as_string(), "cpp::__ESBMC")) {
        cur_state->source.pc++;
        std::string name = id.as_string().substr(5);
        name = name.substr(0, name.find("("));
        run_intrinsic(call, art, name);
        return;
      }
    }

    // Don't run a function call if the guard is false.
    if (!cur_state->guard.is_false()) {
      symex_function_call(deref_code);
    } else {
      cur_state->source.pc++;
    }
  }
  break;

  case OTHER:
    if (!cur_state->guard.is_false()) {
      symex_other();
    }
    cur_state->source.pc++;
    break;

  case CATCH:
    symex_catch();
    break;

  case THROW:
    if (!cur_state->guard.is_false()) {
      if(symex_throw())
        cur_state->source.pc++;
    } else {
      cur_state->source.pc++;
    }
    break;

  case THROW_DECL:
    symex_throw_decl();
    cur_state->source.pc++;
    break;

  case THROW_DECL_END:
    // When we reach THROW_DECL_END, we must clear any throw_decl
    if(stack_catch.size())
    {
      // Get to the correct try (always the last one)
      goto_symex_statet::exceptiont* except=&stack_catch.top();

      except->has_throw_decl=false;
      except->throw_list_set.clear();
    }

    cur_state->source.pc++;
    break;

  default:
    std::cerr << "GOTO instruction type " << instruction.type;
    std::cerr << " not handled in goto_symext::symex_step" << std::endl;
    abort();
  }
}

void
goto_symext::run_intrinsic(const code_function_call2t &func_call,
                           reachability_treet &art, const std::string symname)
{

  if (symname == "__ESBMC_yield") {
    intrinsic_yield(art);
  } else if (symname == "__ESBMC_switch_to") {
    intrinsic_switch_to(func_call, art);
  } else if (symname == "__ESBMC_switch_away_from") {
    intrinsic_switch_from(art);
  } else if (symname == "__ESBMC_get_thread_id") {
    intrinsic_get_thread_id(func_call, art);
  } else if (symname == "__ESBMC_set_thread_internal_data") {
    intrinsic_set_thread_data(func_call, art);
  } else if (symname == "__ESBMC_get_thread_internal_data") {
    intrinsic_get_thread_data(func_call, art);
  } else if (symname == "__ESBMC_spawn_thread") {
    intrinsic_spawn_thread(func_call, art);
  } else if (symname == "__ESBMC_terminate_thread") {
    intrinsic_terminate_thread(art);
  } else if (symname == "__ESBMC_get_thread_state") {
    intrinsic_get_thread_state(func_call, art);
  } else if (symname == "__ESBMC_really_atomic_begin") {
    intrinsic_really_atomic_begin(art);
  } else if (symname == "__ESBMC_really_atomic_end") {
    intrinsic_really_atomic_end(art);
  } else if (symname == "__ESBMC_switch_to_monitor") {
    intrinsic_switch_to_monitor(art);
  } else if (symname == "__ESBMC_switch_from_monitor") {
    intrinsic_switch_from_monitor(art);
  } else if (symname == "__ESBMC_register_monitor") {
    intrinsic_register_monitor(func_call, art);
  } else if (symname == "__ESBMC_kill_monitor") {
    intrinsic_kill_monitor(art);
  } else if (symname == "__ESBMC_realloc") {
    intrinsic_realloc(func_call, art);
  } else if (symname == "__ESBMC_check_stability") {
    intrinsic_check_stability(func_call, art);
  } else if (symname == "__ESBMC_generate_cascade_controllers") {
    // intrinsic_generate_cascade_controllers(func_call, art);
  } else if (symname == "__ESBMC_generate_delta_coefficients") {
    // intrinsic_generate_delta_coefficients(func_call, art);
  } else if (symname == "__ESBMC_check_delta_stability") {
    // intrinsic_check_delta_stability(func_call, art);
  } else {
    std::cerr << "Function call to non-intrinsic prefixed with __ESBMC (fatal)";
    std::cerr << std::endl << "The name in question: " << symname << std::endl;
    std::cerr <<
    "(NB: the C spec reserves the __ prefix for the compiler and environment)"
              << std::endl;
    abort();
  }

  return;
}

void
goto_symext::finish_formula(void)
{

  if (!memory_leak_check)
    return;

  std::list<allocated_obj>::const_iterator it;
  for (it = dynamic_memory.begin(); it != dynamic_memory.end(); it++) {
    // Assert that the allocated object was freed.
    deallocated_obj2tc deallocd(it->obj);
    equality2tc eq(deallocd, true_expr);
    replace_dynamic_allocation(eq);
    it->alloc_guard.guard_expr(eq);
    cur_state->rename(eq);
    target->assertion(it->alloc_guard.as_expr(), eq,
                      "dereference failure: forgotten memory",
                      std::vector<dstring>(), cur_state->source);
    total_claims++;
    remaining_claims++;
  }
}
