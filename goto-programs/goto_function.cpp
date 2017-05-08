/*******************************************************************\

Module: Program Transformation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <i2string.h>
#include <expr_util.h>
#include <location.h>
#include <cprover_prefix.h>
#include <prefix.h>

#include <ansi-c/c_types.h>

#include "goto_convert_class.h"

/*******************************************************************\

Function: goto_convertt::convert_function_call

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_function_call(
  const code_function_callt &function_call,
  goto_programt &dest)
{
  do_function_call(
    function_call.lhs(),
    function_call.function(),
    function_call.arguments(),
    dest);
}

/*******************************************************************\

Function: goto_convertt::do_function_call

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::do_function_call(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  // make it all side effect free
  exprt new_lhs=lhs,
        new_function=function;

  exprt::operandst new_arguments=arguments;

  if(!new_lhs.is_nil())
  {
    remove_sideeffects(new_lhs, dest);
  }

  remove_sideeffects(new_function, dest);

  Forall_expr(it, new_arguments)
  {
    remove_sideeffects(*it, dest);
  }

  // split on the function
  if(new_function.id()=="dereference" ||
     new_function.id()=="implicit_dereference")
  {
    do_function_call_dereference(new_lhs, new_function, new_arguments, dest);
  }
  else if(new_function.id()=="if")
  {
    do_function_call_if(new_lhs, new_function, new_arguments, dest);
  }
  else if(new_function.id()=="symbol")
  {
    do_function_call_symbol(new_lhs, new_function, new_arguments, dest);
  }
  else if(new_function.id()=="NULL-object")
  {
  }
#if 0
  else if(new_function.id()=="member"
          && new_function.has_operands()
          && new_function.op0().statement()=="typeid")
  {
    // Let's create an instruction for typeid

    // First, construct a code with all the necessary typeid infos
    exprt typeid_code("sideeffect");
    typeid_code.set("type", function.op0().op1().op0().find("#cpp_type"));

    typeid_code.statement("typeid");

    typeid_code.operands().push_back(function.op0().op1().op0());
    typeid_code.location() = function.location();

    typeid_code.id("code");
    typeid_code.type()=typet("code");

    // Second, copy to the goto-program
    copy(to_code(typeid_code), OTHER, dest);

    // We must check if the is a exception list
    // If there is, we must throw the exception
    const exprt& exception_list=
      static_cast<const exprt&>(function.op0().find("exception_list"));

    if(exception_list.is_not_nil())
    {
      // Add new instruction throw
      goto_programt::targett t=dest.add_instruction(THROW);
      codet c("cpp-throw");
      c.set("exception_list", exception_list);
      migrate_expr(c, t->code);
      t->location=function.location();
    }
  }
#endif
  else
  {
    err_location(function);
    throw "unexpected function argument: "+new_function.id_string();
  }
}

/*******************************************************************\

Function: goto_convertt::do_function_call_if

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::do_function_call_if(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  if(function.operands().size()!=3)
  {
    err_location(function);
    throw "if expects three operands";
  }

  // case split

  //    c?f():g()
  //--------------------
  // v: if(!c) goto y;
  // w: f();
  // x: goto z;
  // y: g();
  // z: ;

  // do the v label
  goto_programt tmp_v;
  goto_programt::targett v=tmp_v.add_instruction();

  // do the x label
  goto_programt tmp_x;
  goto_programt::targett x=tmp_x.add_instruction();

  // do the z label
  goto_programt tmp_z;
  goto_programt::targett z=tmp_z.add_instruction();
  z->make_skip();

  // y: g();
  goto_programt tmp_y;
  goto_programt::targett y;

  do_function_call(lhs, function.op2(), arguments, tmp_y);

  if(tmp_y.instructions.empty())
    y=tmp_y.add_instruction(SKIP);
  else
    y=tmp_y.instructions.begin();

  // v: if(!c) goto y;
  v->make_goto(y);
  migrate_expr(function.op0(), v->guard);
  v->guard = not2tc(v->guard);
  v->location=function.op0().location();

  unsigned int globals = get_expr_number_globals(v->guard);
  if(globals > 1) {
    exprt tmp = migrate_expr_back(v->guard);
    break_globals2assignments(tmp, tmp_v,lhs.location());
  }

  // w: f();
  goto_programt tmp_w;

  do_function_call(lhs, function.op1(), arguments, tmp_w);

  if(tmp_w.instructions.empty())
    tmp_w.add_instruction(SKIP);

  // x: goto z;
  x->make_goto(z);

  dest.destructive_append(tmp_v);
  dest.destructive_append(tmp_w);
  dest.destructive_append(tmp_x);
  dest.destructive_append(tmp_y);
  dest.destructive_append(tmp_z);
}

/*******************************************************************\

Function: goto_convertt::do_function_call_dereference

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::do_function_call_dereference(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  goto_programt::targett t=dest.add_instruction(FUNCTION_CALL);

  code_function_callt function_call;
  function_call.location()=function.location();
  function_call.lhs()=lhs;
  function_call.function()=function;
  function_call.arguments()=arguments;

  t->location=function.location();
  migrate_expr(function_call, t->code);
}

#include "goto_functions.h"

void goto_functionst::output(
  const namespacet &ns,
  std::ostream& out) const
{
  for(typename function_mapt::const_iterator
      it=function_map.begin();
      it!=function_map.end();
      it++)
  {
    if(it->second.body_available)
    {
      out << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
      out << std::endl;

      const symbolt &symbol=ns.lookup(it->first);
      out << symbol.display_name() << " (" << symbol.name << "):" << std::endl;
      it->second.body.output(ns, symbol.name, out);
    }
  }
}

void goto_functionst::compute_location_numbers()
{
  unsigned nr=0;

  for(typename function_mapt::iterator
      it=function_map.begin();
      it!=function_map.end();
      it++)
    it->second.body.compute_location_numbers(nr);
}

void goto_functionst::compute_incoming_edges()
{
  for(typename function_mapt::iterator
      it=function_map.begin();
      it!=function_map.end();
      it++)
    it->second.body.compute_incoming_edges();
}

void goto_functionst::compute_target_numbers()
{
  for(typename function_mapt::iterator
      it=function_map.begin();
      it!=function_map.end();
      it++)
    it->second.body.compute_target_numbers();
}

void goto_functionst::compute_loop_numbers()
{
  unsigned int num = 0;
  for(typename function_mapt::iterator
      it=function_map.begin();
      it!=function_map.end();
      it++)
    it->second.body.compute_loop_numbers(num);
}
