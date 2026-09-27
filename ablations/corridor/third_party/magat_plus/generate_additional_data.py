"""Extract of `any_additional_data` from the upstream LaGAT repository.

Only this function is needed to build the network; the rest of the upstream
module pulls in that repository's full benchmark harness. See PROVENANCE.md.
"""


def any_additional_data(args):
    additional_data = False
    idx = [None, None, None]
    cur_id = 0
    if args.add_data_cost_to_go:
        additional_data = True
        idx[0] = cur_id
        cur_id += 1
    if args.add_data_greedy_action:
        additional_data = True
        idx[1] = cur_id
        cur_id += 1
    if args.add_data_num_previous_actions is not None:
        additional_data = True
        idx[2] = (cur_id, args.add_data_num_previous_actions)
        cur_id += 1
    return additional_data, idx
