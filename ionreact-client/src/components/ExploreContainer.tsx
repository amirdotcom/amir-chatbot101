import { IonReactRouter } from '@ionic/react-router';
import './ExploreContainer.css';
import { IonIcon, IonLabel, IonRouterOutlet, IonTabBar, IonTabButton, IonTabs } from '@ionic/react';
import { Redirect, Route } from 'react-router';
import HomePage from '../pages/HomePage';
import RadioPage from '../pages/RadioPage';
import LibraryPage from '../pages/LibraryPage';
import SearchPage from '../pages/SearchPage';

import {playCircle, radio, library, search} from 'ionicons/icons';

interface ContainerProps { }

const ExploreContainer: React.FC<ContainerProps> = () => {
  return (
    <IonReactRouter>
    <IonTabs>
      <IonRouterOutlet>
        <Redirect exact path="/" to="/home" />
        {/*
        Use the render method to reduce the number of renders your component will have due to a route change.

        Use the component prop when your component depends on the RouterComponentProps passed in automatically.
      */}
        <Route path="/home" render={() => <HomePage />} exact={true} />
        <Route path="/radio" render={() => <RadioPage />} exact={true} />
        <Route path="/library" render={() => <LibraryPage />} exact={true} />
        <Route path="/search" render={() => <SearchPage />} exact={true} />
      </IonRouterOutlet>

      <IonTabBar slot="bottom">
        <IonTabButton tab="home" href="/home">
          <IonIcon icon={playCircle} />
          <IonLabel>Listen now</IonLabel>
        </IonTabButton>

        <IonTabButton tab="radio" href="/radio">
          <IonIcon icon={radio} />
          <IonLabel>Radio</IonLabel>
        </IonTabButton>

        <IonTabButton tab="library" href="/library">
          <IonIcon icon={library} />
          <IonLabel>Library</IonLabel>
        </IonTabButton>

        <IonTabButton tab="search" href="/search">
          <IonIcon icon={search} />
          <IonLabel>Search</IonLabel>
        </IonTabButton>
      </IonTabBar>
    </IonTabs>
  </IonReactRouter>
  );
};

export default ExploreContainer;
