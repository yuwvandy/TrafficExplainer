# echo ====================Byte-Transformer - ISCX VPN====================
python3 main.py --dataset='iscx-vpn' --baseline='Byte-Transformer'
python3 test.py --dataset='iscx-vpn' --baseline='Byte-Transformer'
python3 explain.py --dataset='iscx-vpn' --baseline='Byte-Transformer' --explanation='byte-level'
python3 explain.py --dataset='iscx-vpn' --baseline='Byte-Transformer' --explanation='saliency map'
python3 explain.py --dataset='iscx-vpn' --baseline='Byte-Transformer' --explanation='random'

# echo ====================Byte-Transformer - ISCX Non-VPN====================
# python3 main.py --dataset='iscx-nonvpn' --baseline='Byte-Transformer'
# python3 test.py --dataset='iscx-nonvpn' --baseline='Byte-Transformer'
# python3 explain.py --dataset='iscx-nonvpn' --baseline='Byte-Transformer' --explanation='byte-level'
# python3 explain.py --dataset='iscx-nonvpn' --baseline='Byte-Transformer' --explanation='saliency map'
# python3 explain.py --dataset='iscx-nonvpn' --baseline='Byte-Transformer' --explanation='random'

# echo ====================Byte-Transformer - ISCX TOR====================
# python3 main.py --dataset='iscx-tor' --baseline='Byte-Transformer'
# python3 test.py --dataset='iscx-tor' --baseline='Byte-Transformer'
# python3 explain.py --dataset='iscx-tor' --baseline='Byte-Transformer' --explanation='byte-level'
# python3 explain.py --dataset='iscx-tor' --baseline='Byte-Transformer' --explanation='saliency map'
# python3 explain.py --dataset='iscx-tor' --baseline='Byte-Transformer' --explanation='random'

# echo ====================Byte-Transformer - ISCX Non-TOR====================
# python3 main.py --dataset='iscx-nontor' --baseline='Byte-Transformer'
# python3 test.py --dataset='iscx-nontor' --baseline='Byte-Transformer'
# python3 explain.py --dataset='iscx-nontor' --baseline='Byte-Transformer' --explanation='byte-level'
# python3 explain.py --dataset='iscx-nontor' --baseline='Byte-Transformer' --explanation='saliency map'
# python3 explain.py --dataset='iscx-nontor' --baseline='Byte-Transformer' --explanation='random'


# echo ====================ET-Bert - ISCX VPN====================
# python3 main.py --dataset='iscx-vpn' --baseline='ET-Bert'
# python3 test.py --dataset='iscx-vpn' --baseline='ET-Bert'

# echo ====================ET-Bert - ISCX nonVPN====================
# python3 main.py --dataset='iscx-nonvpn' --baseline='ET-Bert'
# python3 test.py --dataset='iscx-nonvpn' --baseline='ET-Bert'

# echo ====================ET-Bert - ISCX TOR====================
# python3 main.py --dataset='iscx-tor' --baseline='ET-Bert'
# python3 test.py --dataset='iscx-tor' --baseline='ET-Bert'

# echo ====================ET-Bert - ISCX Non-TOR====================
# python3 main.py --dataset='iscx-nontor' --baseline='ET-Bert'
# python3 test.py --dataset='iscx-nontor' --baseline='ET-Bert'
