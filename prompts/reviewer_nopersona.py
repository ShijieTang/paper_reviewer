from .reviewer_common import TASK, EVAL_CRIT, REVIEW_REQUIRE, OUTPUT_REQUIRE, OUTPUT_FORMAT

reviewer_nopersona = """
###Persona###
You are a paper reviewer. Review the academic paper provided by the user.
""" + TASK + EVAL_CRIT + REVIEW_REQUIRE + OUTPUT_REQUIRE + OUTPUT_FORMAT.replace("<reviewer name>", "Reviewer - No Persona") + """
Now review the following paper:
"""
