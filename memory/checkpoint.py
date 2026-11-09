from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer_context = SqliteSaver.from_conn_string("checkpoints.sqlite")
checkpointer = checkpointer_context.__enter__()