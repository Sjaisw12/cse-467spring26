"""
main.py

Runs the full federated learning pipeline:
- discover/build common feature schema
- initialize server
- run all clients each round
- aggregate on server with Global DP
"""

from __future__ import annotations

from fl_utils import build_feature_schema
from server import FederatedServer

from client1.client1_node import run_client1
from client2.client2_node import run_client2
from client3.client3_node import run_client3
from client4.client4_node import run_client4
from client5.client5_node import run_client5
from client6.client6_node import run_client6
from client7.client7_node import run_client7
from client8.client8_node import run_client8
from client9.client9_node import run_client9
from client10.client10_node import run_client10


def main():
    client_paths = [
        "client1/client_1.xlsx",
        "client2/client_2.xlsx",
        "client3/client_3.xlsx",
        "client4/client_4.xlsx",
        "client5/client_5.xlsx",
        "client6/client_6.xlsx",
        "client7/client_7.xlsx",
        "client8/client_8.xlsx",
        "client9/client_9.xlsx",
        "client10/client_10.xlsx",
    ]

    # Build shared feature schema across all clients
    feature_columns, target_column = build_feature_schema(client_paths)

    print(f"Detected target column: {target_column}")
    print(f"Total shared features: {len(feature_columns)}")

    # Server init
    server = FederatedServer(
        num_features=len(feature_columns),
        global_clip_norm=1.0,
        global_noise_scale=0.05,
        global_learning_rate=1.0,
    )

    NUM_ROUNDS = 5

    for round_idx in range(1, NUM_ROUNDS + 1):
        print(f"\n{'=' * 60}")
        print(f"ROUND {round_idx}")
        print(f"{'=' * 60}")

        global_weights = server.get_global_weights()

        client_payloads = [
            run_client1(feature_columns, global_weights),
            run_client2(feature_columns, global_weights),
            run_client3(feature_columns, global_weights),
            run_client4(feature_columns, global_weights),
            run_client5(feature_columns, global_weights),
            run_client6(feature_columns, global_weights),
            run_client7(feature_columns, global_weights),
            run_client8(feature_columns, global_weights),
            run_client9(feature_columns, global_weights),
            run_client10(feature_columns, global_weights),
        ]

        avg_local_acc = sum(p["local_accuracy"] for p in client_payloads) / len(client_payloads)
        print(f"\nAverage local accuracy before server aggregation: {avg_local_acc:.4f}")

        updated_global = server.aggregate(client_payloads)
        print(f"Global model updated. Weight vector length = {len(updated_global)}")

    print("\nTraining complete.")
    print("Final global weights:")
    print(server.get_global_weights())


if __name__ == "__main__":
    main()