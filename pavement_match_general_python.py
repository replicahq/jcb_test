import geopandas as gpd
from shapely.geometry import LineString
from shapely import ops
import rtree.index as rindex
import pyproj
import pandas as pd
import os 
import math
from difflib import SequenceMatcher
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process geospatial data.")
    parser.add_argument("mainfold", help="Path to the main folder")
    parser.add_argument("--use_pavement", default="use_pavement.gpkg", 
                        help="Input file for pavement data")
    parser.add_argument("--base_streets", default="base_streets.gpkg", 
                        help="Input file for base streets data")
    parser.add_argument("target_length", type=float, help="Target length")
    parser.add_argument("max_dist", type=float, help="Maximum distance")
    return parser.parse_args()

def delFields(gdf, fieldlist):
    fields = [c for c in gdf.columns if c not in fieldlist]
    gdf.drop(columns=fields, inplace=True)

def interpolate_line(line, distances):
    total_length = line.length
    pts = []
    for d in distances:
        pt = line.interpolate(d * total_length)
        pts.append(pt)
    return pts

def get_utm_zone_epsg(latitude, longitude):
    utm = int((longitude + 180) / 6) + 1

    if latitude >= 0:
        epsg = f'EPSG:326{utm:02d}'
    else:
        epsg = f'EPSG:327{utm:02d}'
    return epsg

def convert_to_decimal_degrees(coordinates, crs):
    proj = pyproj.Transformer.from_crs(pyproj.CRS(crs),
                                       pyproj.CRS.from_epsg(4326),
                                       always_xy=True)
    lon, lat = proj.transform(coordinates[0], 
                              coordinates[1])
    return lat, lon

def extract_first_point_coords(geometry, crs):
    first_line = geometry.geoms[0]
    first_point = first_line.coords[0]
    return convert_to_decimal_degrees(first_point, crs)

def prep_gdf(gdf, epsg):
    if not gdf.geometry.is_valid.all():
        gdf = gdf[gdf.geometry.is_valid].buffer(0)
    if gdf.sindex is None:
        gdf.sindex = gdf.sindex.reset_index(drop=True)
    gdf = gdf.to_crs(epsg)
    return gdf

def get_azimuth(line):
    #print(line.geom_type)
    if line.geom_type == 'LineString':
        # Extract coordinates of the first and last points
        first_point = line.coords[0]
        last_point = line.coords[-1]

        # Calculate azimuth using the first and last points
        radian = math.atan2((last_point[1] - first_point[1]), 
                            (last_point[0] - first_point[0]))
        degrees = math.degrees(radian)

        return degrees
    else:
        return 999
    
def remove_suffix_and_prefix(text, prefixes, suffixes):
    if pd.isnull(text):
        return text
    text = text.replace("'", '').replace('-', ' ')
    suffix_pattern = r'\b(?:{})\b'.format('|'.join(suffixes))
    prefix_pattern = r'^(?:{})\s'.format('|'.join(prefixes))

    pattern = r'({})|({})'.format(suffix_pattern, prefix_pattern)
    text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
    return text

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def lev_dist(a, b):
    m = len(a)
    n = len(b)

    # Create a matrix to store the distances
    distances = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column
    for i in range(m + 1):
        distances[i][0] = i
    for j in range(n + 1):
        distances[0][j] = j

    # Calculate the distances
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            distances[i][j] = min(
                distances[i - 1][j] + 1,  # deletion
                distances[i][j - 1] + 1,  # insertion
                distances[i - 1][j - 1] + cost,  # substitution
            )

    # Return the final distance
    return distances[m][n]

def save_and_print_results(comps, mainfold, id_col):
    comps.sort_values(id_col, ascending=True, inplace=True)
    print(comps.iloc[3])
    comps.to_csv(os.path.join(mainfold, 
                              'comparison_first_draft.csv'), 
                              index=False)

def lev_sim(s1, s2):
    # Calculate the Levenshtein distance
    distance = lev_dist(s1, s2)
    # Convert the distance to a similarity score
    similarity = 1 / (1 + distance)
    return similarity

def main():
    args = parse_args()
    mainfold = args.mainfold
    target_length = args.target_length
    max_dist = args.max_dist
    infile = args.use_pavement
    insts = args.base_streets

    usepave = os.path.join(mainfold, infile)
    base_sts = os.path.join(mainfold, insts)

    dtypes = {'stableEdgeId': str}

    gdf = gpd.read_file(usepave)

    point = extract_first_point_coords(gdf['geometry'].iloc[0], gdf.crs)
    utm = get_utm_zone_epsg(point[0], point[1])

    basegdf = gpd.read_file(base_sts, dtype=dtypes)

    gdf = prep_gdf(gdf, utm)
    basegdf = prep_gdf(basegdf, utm)

    keep_keywords = ['ID', 'STREET', 'NAME', 
                    'FUNCTIONAL', 'CLASSIFICATION']

    keeps = [col for col in gdf.columns if 
            any(keyword in col.upper() for keyword in keep_keywords)] + \
                ['geometry']

    func_col = next((c for c in keeps if 'FUNCTIONAL' in c.upper() \
                            or 'CLASSIFICATION' in c.upper()), None)

    gdf = gdf.loc[~gdf[func_col].isna()]

    remove_class = ['Bikeway', 'Centroid Connector']

    gdf = gdf.loc[~gdf[func_col].isin(remove_class)]

    id_col = next((c for c in keeps if 'ID' in c.upper() \
                and gdf[c].nunique() == len(gdf)), None)

    delFields(gdf, keeps)

    basegdf['heading'] = basegdf['geometry'].apply(get_azimuth)

    short_df = gdf.loc[gdf['geometry'].length < target_length]
    gdf = gdf.loc[gdf['geometry'].length >= target_length]

    gdf = gdf.explode(index_parts=True)
    gdf['heading'] = gdf['geometry'].apply(get_azimuth)

    if len(short_df) > 0:
        short_df.to_csv(os.path.join(mainfold, 'short_segs.csv'), 
                        index=False)

    distances = [i / 100 for i in range(48, 53)]

    # Replicate rows for each interpolated point
    gdf_expanded = gdf.loc[gdf.index.repeat(len(distances))]
    gdf_expanded['distances'] = distances * len(gdf)

    # Interpolate points directly for all rows
    gdf_expanded['geometry'] = gdf_expanded.apply(lambda row: \
        interpolate_line(row['geometry'], 
                        [row['distances']])[0], axis=1)

    # Create the GeoDataFrame
    inter_pts = gpd.GeoDataFrame(gdf_expanded[[id_col, 'geometry']], crs=gdf.crs)

    idx = rindex.Index()
    for i, line in enumerate(basegdf.geometry):
        idx.insert(i, line.bounds)

    nearest = []
    for pt in inter_pts.geometry:

        pt_coords = list(pt.coords)[0]
        pt_buff = ops.transform(lambda x, y, 
                                z=None: (x, y), pt).buffer(max_dist, 
                                                        cap_style=3, 
                                                            join_style=2)
        candidates = list(idx.intersection(pt_buff.bounds))
        indices = sorted(candidates, key=lambda i: \
                        pt.distance(basegdf.geometry.iloc[i]))[:2]
        ids = basegdf.stableEdgeId.iloc[indices].tolist() if indices else []
        lines = basegdf.geometry.iloc[indices].tolist() if indices else [None, None]
        lines = [line if line is not None else LineString() for line in lines]
        distances = [pt.distance(line) for line in lines]

        filt_indices = [i for i, dist in enumerate(distances) if dist <= max_dist]
        filt_ids = [ids[i] for i in filt_indices]
        filt_distances = [distances[i] for i in filt_indices]
        nearest.append({'point': pt, 
                        'stableEdgeId': filt_ids, 
                        'distances': filt_distances})

    nearest = pd.DataFrame(nearest)
    nearest = nearest.explode(['stableEdgeId', 'distances'], 
                            ignore_index=True)
    result = pd.merge(nearest, 
                    inter_pts[['geometry', id_col]], 
                    left_on='point', right_on='geometry', 
                    how='inner')

    del nearest

    result = result.drop(['point', 'geometry'], axis=1)

    just_sts = basegdf[['stableEdgeId', 'streetName', 'highway', 
                        'heading']].drop_duplicates()

    just_sts.rename(columns={'heading': 'base_heading'}, inplace=True)

    merged_df = pd.merge(result, 
                        just_sts[['stableEdgeId', 'streetName', 
                                'highway', 'base_heading']], 
                        on='stableEdgeId', 
                        how='left')

    street_cols = [c for c in gdf.columns if 'STREET' in c.upper() \
                or 'NAME' in c.upper()]

    merged_df = pd.merge(merged_df, gdf[[id_col, street_cols[0], func_col, 'heading']],
                        on=id_col,
                        how='left')

    merged_df = merged_df.loc[~merged_df['stableEdgeId'].isna()]

    suffixes = ['Street', 'St', 'Terrace', 'Ter', 'Court', 'Ct', 'Place',
                'Pl', 'Circle', 'Cir', 'Road', 'Rd', 'Boulevard', 'Blvd',
                'Avenue', 'Ave', 'Drive', 'Dr', 'Lane', 'Ln', 'Parkway',
                'Pkwy', 'Plaza', 'Plz', 'Trafficway', 'Trfy', 'Cutoff',
                'Ctof', 'Way', 'Wy', 'Highway', 'Hwy', 'NW', 'SW', 'SE',
                'NE', 'N', 'S', 'W', 'E', 'North', 'Northeast', 'Northwest',
                'South', 'Southeast', 'Southwest', 'East', 'West', 'Landing',
                'Ldg', 'Row', 'Rw', 'Center', 'Ctr', 'Isle', 'Pass', 'Trail',
                'Trl']

    prefixes = ['NW', 'SW', 'SE', 'NE', 'N', 'S', 'W', 'E', 'North', 
                'Northeast', 'Northwest', 'South', 'Southeast', 'Southwest', 
                'East', 'West']

    numbermap = {'1ST': 'FIRST', '2ND': 'SECOND', '3RD': 'THIRD', '4TH':
                'FOURTH', '5TH': 'FIFTH', '6TH': 'SIXTH', '7TH': 'SEVENTH',
                '8TH': 'EIGHTH', '9TH': 'NINTH'}

    merged_df['transStNm'] = merged_df['streetName'].apply(remove_suffix_and_prefix,
                                                        prefixes=prefixes,
                                                        suffixes=suffixes)

    del merged_df['streetName']

    street_cols = [c for c in keeps if 'STREET' in c.upper() \
                or 'NAME' in c.upper()]
    
    if not street_cols:
        print("No 'STREET' or 'NAME' columns found. Skipping street-related operations.")
        comps = merged_df.copy()
        save_and_print_results(comps, mainfold, id_col)
        return 

    merged_df[street_cols] = merged_df[street_cols].\
        map(lambda x: numbermap.get(x, x))

    merged_df[street_cols[0]] = merged_df[street_cols[0]].apply(remove_suffix_and_prefix,
                                                        prefixes=prefixes,
                                                        suffixes=suffixes)

    merged_df[street_cols[0]] = merged_df[street_cols[0]].str.upper()
    merged_df['transStNm'] = merged_df['transStNm'].str.upper()

    merged_df['similarity'], merged_df['lev_sim'] = zip(*merged_df.apply(lambda row: (
        similar(str(row[street_cols[0]]), 
                str(row['transStNm'])) if pd.notnull(row['transStNm']) else 0,
        lev_sim(str(row[street_cols[0]]), 
                str(row['transStNm'])) if pd.notnull(row['transStNm']) else 0
    ), axis=1))

    comps = merged_df.groupby(['stableEdgeId', id_col, 
                            street_cols[0], 'highway', 'transStNm', 
                            func_col]).mean().reset_index()

    comps['similarity'] = comps['similarity'] + 0.01
    comps['score'] = comps['distances'] / comps['similarity']

    comps['rank'] = comps.groupby([id_col])['score'].rank(ascending=True,
                                                            method='min')

    comps = comps.loc[comps['rank'] == 1]
    comps['distances'] = comps['distances'].astype('float')

    floats = [c for c in comps.columns if comps[c].dtype == 'float']

    for f in floats:
        comps[f] = comps[f].round(4)

    save_and_print_results(comps, mainfold, id_col)

if __name__ == "__main__":
    main()
