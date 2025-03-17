from open_spiel.python.rl_environment import Environment

from eval.sampler import HandDealer


def test_dealer():
    dhs = HandDealer()
    env = Environment("tiny_bridge_4p", chance_event_sampler=dhs)
    dhs.seed_deal([15, 11, 24, 8])
    env.reset()
    assert (str(env.get_state)) == "W:SKHJ N:SQHQ E:SAHA S:SJHK"
