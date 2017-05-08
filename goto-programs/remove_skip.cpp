/*******************************************************************\

Module: Program Transformation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "remove_skip.h"

/*******************************************************************\

Function: is_skip

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static bool is_skip(goto_programt::instructionst::iterator it)
{
  if (it->is_skip())
    return true;

  if (it->is_goto())
  {
    if (is_constant_bool2t(it->guard) &&
        !to_constant_bool2t(it->guard).constant_value)
      return true;

    if (it->targets.size()!=1)
      return false;

    goto_programt::instructionst::iterator next_it=it;
    next_it++;

    return it->targets.front()==next_it;
  }

  if(it->is_other())
  {
    return is_nil_expr(it->code);
  }

  return false;
}

/*******************************************************************\

Function: remove_skip

  Inputs:

 Outputs:

 Purpose: remove unnecessary skip statements

\*******************************************************************/

void remove_skip(goto_programt &goto_program)
{
  typedef std::map<goto_programt::targett, goto_programt::targett> new_targetst;
  new_targetst new_targets;

  // remove skip statements

  for(goto_programt::instructionst::iterator
      it=goto_program.instructions.begin();
      it!=goto_program.instructions.end();)
  {
    goto_programt::targett old_target=it;

    // for collecting labels
    std::list<irep_idt> labels;

    while(is_skip(it))
    {
      // don't remove the last skip statement,
      // it could be a target
      if(it==--goto_program.instructions.end())
        break;

      // save labels
      labels.splice(labels.end(), it->labels);
      it++;
    }

    goto_programt::targett new_target=it;

    // save labels
    it->labels.splice(it->labels.begin(), labels);

    if(new_target!=old_target)
    {
      while(new_target!=old_target)
      {
        // remember the old targets
        new_targets[old_target]=new_target;
        old_target=goto_program.instructions.erase(old_target);
      }
    }
    else
      it++;
  }

  // adjust gotos

  Forall_goto_program_instructions(i_it, goto_program)
    if(i_it->is_goto())
    {
      for(goto_programt::instructiont::targetst::iterator
          t_it=i_it->targets.begin();
          t_it!=i_it->targets.end();
          t_it++)
      {
        new_targetst::const_iterator
          result=new_targets.find(*t_it);

        if(result!=new_targets.end())
          *t_it=result->second;
      }
    }

  // remove the last skip statement unless it's a target
  goto_program.compute_incoming_edges();

  if(!goto_program.instructions.empty() &&
     is_skip(--goto_program.instructions.end()) &&
     !goto_program.instructions.back().is_target())
    goto_program.instructions.pop_back();
}

/*******************************************************************\

Function: remove_skip

  Inputs:

 Outputs:

 Purpose: remove unnecessary skip statements

\*******************************************************************/

void remove_skip(goto_functionst &goto_functions)
{
  Forall_goto_functions(f_it, goto_functions)
    remove_skip(f_it->second.body);
}
