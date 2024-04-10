# Concept Shallot Explorer

The Shallot Explorer is a module of the BeyonCE (Beyond Common Eclipsers) package used to explore the large parameter space of eclipsing disc systems. From a light curve we have the hard limit of the duration of the eclipse, $\Delta_\mathrm{ecl}$. From there we can think of infinite discs (projected ellipses) that have intercepts with the beginning and end of the eclipse.

The Shallot Explorer produces a 3-D grid with the centre of the projected ellipse (this is also the centre of the disc as we assume it is azimuthally symmetric) at a location $\delta x$, $\delta y$ and we have a third dimension called the radius factor $f_R$.

It is important to realise that all the "spatial" dimensions obtained from the Shallot Explorer are converted to the temporal domain, and to generalise the grid we express the dimensions in units of $\Delta_\mathrm{ecl}$.

## Grid Visualisation
### Dimensions

1. The grid has an $x$ dimension corresponding to $\delta x$, which is the shift that the centre of the projected ellipse has w.r.t. the centre of the eclipse.
2. The grid has a $y$ dimension corresponding to $\delta y$, which is the shift that the centre of the projected ellipse has w.r.t. the centre of the eclipse. This is commonly known as the impact parameter.
3. The grid has a $z$ dimension corresponding to $f_R$, the radius factor. 

This factor stems from the fact that for any given ($\delta x, \delta y$) there is an infinite number of ellipses that intersect at the boundaries ($\pm \frac{\Delta_\mathrm{ecl}}{2}, 0$). If we find start the smallest disc radius, we can describe the rest of the ellipses by scaling the radius (which in turn affects the inclination of the disc) by some factor $f_R > 1$. It should also be noted that one can scale the radius of the disc in the direction of the semi-major axis or the semi-minor axis of the projected ellipse.

### 