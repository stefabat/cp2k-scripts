#!/bin/bash
 
cutoffs="150 200 250 300 350 400 450 500"
 
project=
input_file=$project.inp
output_file=$project.out
 
plot_file=cutoff_convergence.ssv
 
#echo "Cutoff (Ry)   rho(G)        E_Hartree (Ha)  E_xc (Ha)    E_tot (Ha)    E_tot/atom (Ha)" >> $plot_file
printf "%10s  %14s  %18s  %18s  %18s  %18s" "Cutoff (Ry)" "rho_tot(G)" "E_Hartree (Ha)" "E_xc (Ha)" "E_tot (Ha)" "E_tot/atom (Ha)" > $plot_file
grid_header=true
for ii in $cutoffs ; do
    work_dir=cutoff_${ii}Ry
    
    # grep data
    rho=$(grep -e '^[ \t]*Total charge density g-space grids' $work_dir/$output_file | awk '{print $6}')
    e_core=$(grep -e '^[ \t]*Core Hamiltonian energy' $work_dir/$output_file | awk '{print $4}')
    e_h=$(grep -e '^[ \t]*Hartree energy' $work_dir/$output_file | awk '{print $3}')
    e_xc=$(grep -e '^[ \t]*Exchange-correlation energy' $work_dir/$output_file | awk '{print $3}')
    e_tot=$(grep -e '^[ \t]*Total energy' $work_dir/$output_file | awk '{print $3}')
    n_atoms=$(grep -i '\- atoms' $work_dir/$output_file | awk '{print $3}')
    n_grids=$(grep -e '^[ \t]*QS| Number of grid levels:' $work_dir/$output_file | awk '{print $6}')
    
    # calculate quantities per atom
    e_tot_atom=$(awk "BEGIN {printf \"%.10f\", $e_tot/$n_atoms}")

    if $grid_header ; then
        for ((igrid=1; igrid <= n_grids; igrid++)) ; do
            printf "    NG grid %d" $igrid >> $plot_file
        done
        printf "\n" >> $plot_file
        grid_header=false
    fi

    printf "%10u  %14.10f  %18.10f  %18.10f  %18.10f  %18.10f" $ii $rho $e_h $e_xc $e_tot $e_tot_atom >> $plot_file
    
    for ((igrid=1; igrid <= n_grids; igrid++)) ; do
        grid=$(grep -e '^[ \t]*count for grid' $work_dir/$output_file | awk -v igrid=$igrid '(NR == igrid){print $5}')
        printf " %12d" $grid >> $plot_file
    done
    printf "\n" >> $plot_file
done
