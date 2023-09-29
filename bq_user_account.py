import sys
from google.cloud import bigquery

def set_permissions_for_user(user_email, datasets, project_id='replica-customer'):
    # Initialize the BigQuery client
    client = bigquery.Client()

    for dataset_id in datasets:
        dataset_ref = client.dataset(dataset_id, project=project_id)

        dataset = client.get_dataset(dataset_ref)

        # Add a new access entry for the user
        access_entries = dataset.access_entries
        access_entries.append(
            bigquery.AccessEntry(
                role='READER',
                entity_type='userByEmail',
                entity_id=user_email
            )
        )

        # Update the dataset with the new access entries
        dataset.access_entries = access_entries
        client.update_dataset(dataset, ['access_entries'])

        print(f'Permissions set for dataset: {dataset_id} for user: {user_email}')

    print('Dataset-level permissions have been set for the user.')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <user_email>")
        sys.exit(1)

    user_email = sys.argv[1]

    datasets = ['cal_nev', 'hawaii', 'great_lakes', 'north_atlantic', 'north_central',
                'northeast', 'northwest', 'mid_atlantic', 'south_atlantic',
                'south_central', 'southwest', 'Geos', 'reference']

    set_permissions_for_user(user_email, datasets)
