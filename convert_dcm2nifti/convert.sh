
# /vol/bitbucket/rh2515/dcm2niix/build/bin/dcm2niix -o /vol/bitbucket/rh2515/CT_Colonography /vol/bitbucket/bkainz/TCIA/CT\ COLONOGRAPHY/1.3.6.1.4.1.9328.50.4.0001

search_dir=/vol/bitbucket/bkainz/TCIA/CT\ COLONOGRAPHY
out_dir=/vol/bitbucket/rh2515/CT_Colonography

for dcm_folder in "$search_dir"/*
do
  echo "Converting ${dcm_folder}..."
  out_folder=${out_dir}/$(basename -- "$dcm_folder")
  echo ${out_folder}
  mkdir -p $out_folder
  output=$(/vol/bitbucket/rh2515/dcm2niix/build/bin/dcm2niix -o "${out_folder}" -f %j "${dcm_folder}")


  if [[ $output == *"incompatible with NIfTI format"* ]]; then
    echo "Bad conversion: $(basename -- "$dcm_folder")"
  fi

done
